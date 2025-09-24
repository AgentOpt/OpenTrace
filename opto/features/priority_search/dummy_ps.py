from opto.features.priority_search.priority_search import PrioritySearch, Samples, ModuleCandidate
import copy
from opto.trainer.utils import  async_run
import time
from opto.trainer.algorithms.basic_algorithms import batchify
from opto.features.priority_search.utils import set_module_parameters, remap_update_dict, create_module_from_update_dict, is_module_copy
REAL_STRING = """Here are the additional instructions to help the agent solve the task:

- Always proactively offer to find item IDs for the user if they don't have them readily available.

- **Important: Only one modification is allowed per order. Remind the customer of this rule at the beginning of the conversation, *before* proceeding with any modifications. If the user attempts to modify additional items or orders after already modifying one, immediately reiterate the one-modification-per-order rule. This applies even if the initial modification request was not fully completed. For example, if a user starts an exchange but then changes their mind and wants a return, reiterate that only one action (exchange OR return) is allowed per order.**

- When exchanging or modifying items, first confirm the order ID and authenticate the user.

- **When authenticating, if the email lookup fails, immediately ask for first name, last name, and zip code. Avoid going back and forth between authentication methods.**

- **Before suggesting replacement items, check if the order status is "pending". Only suggest replacement items for pending orders. Before suggesting replacement items, remind the customer that only one modification per order is allowed and ensure they are prepared to list ALL desired changes at once.** Then, use the available tools to check the current availability of those items, and their properties, to prevent suggesting unavailable or unsuitable replacements.

- **If `get_product_details` calls are consistently unsuccessful, pause and confirm the user's desired product features and item IDs. Double-check if the product exists. If issues persist after confirming the information, consider transferring the user to a human agent.**

- When offering choices for exchanges, provide at least 2-3 relevant options whenever possible.

- **After authenticating the user and confirming the order ID, retrieve the user's available payment methods *before* proceeding with replacement item suggestions. This will save time later in the exchange process.**

- After the user confirms the items for exchange and the payment method, **immediately proceed to finalize the exchange/modification using the appropriate tools.**

- Remind the user to confirm they have provided all items to be exchanged or modified **as only one modification is allowed per order.** If the user attempts to modify additional items after already modifying one, inform them "Only one modification is allowed per order. If you wish to modify more items in the future, you will have to create another order."

- **Before looking up replacement items, remind the customer only one modification per order is allowed. Ensure they are prepared to list ALL desired changes at once.**

- Common Pitfall: Avoid repeatedly asking for item IDs if the user indicates they need help finding them.

- Best Practice: Briefly summarize the exchange/modification details (original item and replacement item details including IDs, colors, sizes, etc. as applicable) before confirming with the user and making any tool calls to retrieve replacement items. **Before retrieving replacement items, CONFIRM all items the user wants to modify in this single allowed modification.**

- **Workflow Tip:** When a user says they want to modify a *t-shirt* order, assume they want to modify the t-shirt item itself.

- **Efficiency Tip:** Filter `get_user_details` order list to directly show orders with "pending" status and the requested product type (e.g., t-shirts). **If the user doesn't know the order ID for a return or exchange, immediately filter the `get_user_details` order list by item name (e.g., "t-shirt", "water bottle") to efficiently locate the correct order, even if the user has specified the product type. If the user specifies the item names, use these to filter the user's order history.** For exchanges and modifications, the agent should immediately filter for pending orders of the specific item mentioned by the user.

- **Important:** If the user changes their mind during an exchange/return and attempts to modify an additional item, reiterate the one-modification-per-order rule.

- **Mixed Request Handling:** If the user has a mixed request (e.g., modifying a t-shirt in an order that contains other items like an espresso machine), acknowledge all items in the order but focus on the t-shirt modification first.


- **Exchange Completion:** When a price difference occurs during an exchange, clearly explain the difference and ask how the user would like to pay the remaining amount (if applicable).

- **Multiple Order Modification Request:** If a user requests to modify more than one order, inform them that only one order can be modified, and that you can only modify one order for them. For example: "I understand you want to modify multiple orders. However, only one order modification can happen. I can only modify a single order for you at this time."

"""
class DummyPrioritySearch(PrioritySearch):
    """ Disable the optimizer where llm is used to propose parameters. """
    def propose(self,
                samples : Samples,
                verbose : bool = False,
                **kwargs):
        """ Analyzing samples and propose new parameters using self.optimizer. An independent optimizer is used for the minibatch generated by one agent and generates n_proposals proposals.

        Args:
            samples (Samples): Samples collected by the exploration candidates. If None, the agent's parameters are returned without updating.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments that may be used by the implementation.

        Returns:
            candidates (list of ModuleCandidate): A list of proposed candidates for the next iteration.
        """
        print("--- Proposing new parameters...") if verbose else None
        assert isinstance(samples, Samples), "samples must be an instance of Samples."
        samples = samples.samples  # list of BatchRollout objects
        n_proposals = self.num_proposals  # number of proposals to generate per optimizer
        
        # Associate each BatchRollout with self._exploration_candidates
        matched_candidates_and_samples = self.match_candidates_and_samples(self._exploration_candidates, samples)
        # NOTE len(matched_candidates_and_samples) <= len(self._exploration_candidates) since some exploration candidates might be duplicated.
        candidate_batchrollouts_list = [ (k,b) for k, v in matched_candidates_and_samples.items() for b in v]
        assert len(samples) == len(candidate_batchrollouts_list), "All samples must be associated with exploration candidates."
        n_batches = len(samples)  # number of batch rollouts in the samples

        # need to copy optimizer for the n_batches
        def _backward(n):
            candidate, rollouts = candidate_batchrollouts_list[n]
            optimizer = candidate.optimizer or self.optimizer
            # Create a copy of the optimizer to avoid modifying the original one and to allow parallel execution
            optimizer = copy.deepcopy(optimizer)
            optimizer.parameters = rollouts.module.parameters()  # set the optimizer's parameters to the proposal's parameters
            targets = [r.target for r in rollouts]
            feedbacks = [r.feedback for r in rollouts]
            # batchify the targets and feedbacks
            target = batchify(*targets)
            feedback = batchify(*feedbacks).data  # str
            # standard optimizer step
            optimizer.zero_feedback()  # reset the optimizer's feedback
            optimizer.backward(target, feedback)  # compute the gradients based on the targets and feedbacks
            return optimizer

        args_list = [(n,) for n in range(n_batches)]
        optimizers = async_run([_backward]*n_batches,  # run the optimizer step for each agent in parallel
                                 args_list=args_list,
                                 max_workers=self.num_threads,  # use the number of threads specified in the class
                                 description='Backward')
        assert len(optimizers) == n_batches, "Number of optimizers must match number of batch rollouts."
        # need to copy optimizer for the n_proposals
        # NOTE when optimizer is deepcopied, its parameters are not copied.
        optimizers = [copy.deepcopy(o) for o in optimizers ] * n_proposals  # repeat args_list n_proposals times
        assert len(optimizers) == n_batches * n_proposals, "Number of optimizers must match number of batch rollouts times number of proposals."

        # For each optimizer, containing the backward feedback, we call it n_proposals times to get the proposed parameters.
        def _step(n):
            # return a dummy update dict without calling the optimizer
            update_dict = {}
            for param in self.agent.parameters():
                # values are random strings using time.time()
                # Make the random_string very long
                # update_dict[param] = "random_string_" + str(time.time()) 
                update_dict[param] = REAL_STRING+ str(time.time())
            update_dict = remap_update_dict(self.agent, update_dict)  # remap the update dict to the agent's parameters
            return update_dict  # return the proposed parameters

        args_list = [(n,) for n in range(n_batches*n_proposals)]
        update_dicts = async_run([_step]*n_batches*n_proposals,  # run the optimizer step for each agent in parallel
                                  args_list=args_list,
                                  max_workers=self.num_threads,  # use the number of threads specified in the class
                                  description=f"Calling optimizers: Generating {n_proposals} proposals for each of {n_batches} batches",)

        # update_dicts is a list of dicts of length n_batches * n_proposals
        # Create ModuleCandidate objects for each proposed update_dict that is non-trivial
        candidates = [ModuleCandidate(self.agent, update_dict, optimizer)
                        for update_dict, optimizer in zip(update_dicts, optimizers) if update_dict is not None]  # filter out None updates
        return candidates