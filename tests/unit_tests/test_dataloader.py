from opto.trainer.loader import DataLoader



def run_for_loop(dataloader):
    print('Running for-loop')

    counter = 0
    for i, (inputs, infos) in enumerate(dataloader):

        print(f"Inputs: {inputs}, Infos: {infos}")

        if i == 0:
            assert inputs == [1, 2], f"First batch should contain inputs 1 and 2. Get: {inputs}"
            assert infos == ['a', 'b'], f"First batch should contain infos 'a' and 'b'. Get: {infos}"
        elif i == 1:
            assert inputs == [3, 4], f"Second batch should contain inputs 3 and 4. Get: {inputs}"
            assert infos == ['c', 'd'], f"Second batch should contain infos 'c' and 'd'. Get: {infos}"
        elif i == 2:
            assert inputs == [5, 1], f"Third batch should contain inputs 5 and 1. Get: {inputs}"
            assert infos == ['e', 'a'], f"Third batch should contain infos 'e' and 'a'. Get: {infos}"
        counter += 1

    assert counter == 3, f"Should have 3 batches in total. Get: {counter}"

    # Make sure it can be iterated multiple times
    counter = 0
    for i, (inputs, infos) in enumerate(dataloader):

        print(f"Inputs: {inputs}, Infos: {infos}")

        if i == 0:
            assert inputs == [1, 2], f"First batch should contain inputs 1 and 2. Get: {inputs}"
            assert infos == ['a', 'b'], f"First batch should contain infos 'a' and 'b'. Get: {infos}"
        elif i == 1:
            assert inputs == [3, 4], f"Second batch should contain inputs 3 and 4. Get: {inputs}"
            assert infos == ['c', 'd'], f"Second batch should contain infos 'c' and 'd'. Get: {infos}"
        elif i == 2:
            assert inputs == [5, 1], f"Third batch should contain inputs 5 and 1. Get: {inputs}"
            assert infos == ['e', 'a'], f"Third batch should contain infos 'e' and 'a'. Get: {infos}"
        counter += 1

    assert counter == 3, f"Should have 3 batches in total. Get: {counter}"


def run_next(dataloader):

    inputs, infos = next(dataloader)
    print('Running next()')
    print(f"Inputs: {inputs}, Infos: {infos}")

    assert inputs == [1, 2], f"First batch should contain inputs 1 and 2. Get: {inputs}"
    assert infos == ['a', 'b'], f"First batch should contain infos 'a' and 'b'. Get: {infos}"

    inputs, infos = next(dataloader)
    print(f"Inputs: {inputs}, Infos: {infos}")

    assert inputs == [3, 4], f"Second batch should contain inputs 3 and 4. Get: {inputs}"
    assert infos == ['c', 'd'], f"Second batch should contain infos 'c' and 'd'. Get: {infos}"

    inputs, infos = next(dataloader)
    print(f"Inputs: {inputs}, Infos: {infos}")

    assert inputs == [5, 1], f"Third batch should contain inputs 5 and 1. Get: {inputs}"
    assert infos == ['e', 'a'], f"Third batch should contain infos 'e' and 'a'. Get: {infos}"

    try:
        next(dataloader)
    except StopIteration:
        print("No more data to iterate over, as expected.")

    inputs, infos = next(dataloader)
    print('Running next()')
    print(f"Inputs: {inputs}, Infos: {infos}")

    assert inputs == [1, 2], f"First batch should contain inputs 1 and 2. Get: {inputs}"
    assert infos == ['a', 'b'], f"First batch should contain infos 'a' and 'b'. Get: {infos}"

    inputs, infos = next(dataloader)
    print(f"Inputs: {inputs}, Infos: {infos}")

    assert inputs == [3, 4], f"Second batch should contain inputs 3 and 4. Get: {inputs}"
    assert infos == ['c', 'd'], f"Second batch should contain infos 'c' and 'd'. Get: {infos}"

    inputs, infos = next(dataloader)
    print(f"Inputs: {inputs}, Infos: {infos}")

    assert inputs == [5, 1], f"Third batch should contain inputs 5 and 1. Get: {inputs}"
    assert infos == ['e', 'a'], f"Third batch should contain infos 'e' and 'a'. Get: {infos}"

    try:
        next(dataloader)
    except StopIteration:
        print("No more data to iterate over, as expected.")


def run_sample(dataloader):

    print('Running sample()')
    inputs, infos = dataloader.sample()
    assert inputs == [1, 2], f"First sample should contain inputs 1 and 2. Get: {inputs}"
    assert infos == ['a', 'b'], f"First sample should contain infos 'a' and 'b'. Get: {infos}"
    assert dataloader._exhausted == False, "Dataloader should be marked as exhausted after sampling all data."
    inputs, infos = dataloader.sample()
    assert inputs == [3, 4], f"Second sample should contain inputs 3 and 4. Get: {inputs}"
    assert infos == ['c', 'd'], f"Second sample should contain infos 'c' and 'd'. Get: {infos}"
    assert dataloader._exhausted == False, "Dataloader should be marked as exhausted after sampling all data."
    inputs, infos = dataloader.sample()
    assert inputs == [5, 1], f"Third sample should contain inputs 5 and 1. Get: {inputs}"
    assert infos == ['e', 'a'], f"Third sample should contain infos 'e' and 'a'. Get: {infos}"
    assert dataloader._exhausted == False, "Dataloader should be marked as exhausted after sampling all data."

    print('Calling sample does not exhaust the dataloader.')

    # check that it can be sampled again
    inputs, infos = dataloader.sample()
    assert inputs == [2, 3], f"First sample should contain inputs 2 and 3. Get: {inputs}"
    assert infos == ['b', 'c'], f"First sample should contain infos 'b' and 'c'. Get: {infos}"
    assert dataloader._exhausted == False, "Dataloader should be marked as exhausted after sampling all data."
    inputs, infos = dataloader.sample()
    assert inputs == [4, 5], f"Second sample should contain inputs 4 and 5. Get: {inputs}"
    assert infos == ['d', 'e'], f"Second sample should contain infos 'd' and 'e'. Get: {infos}"
    assert dataloader._exhausted == False, "Dataloader should be marked as exhausted after sampling all data."
    inputs, infos = dataloader.sample()
    assert inputs == [1, 2], f"Third sample should contain inputs 1 and 2. Get: {inputs}"
    assert infos == ['a', 'b'], f"Third sample should contain infos 'a' and 'b'. Get: {infos}"
    assert dataloader._exhausted == False, "Dataloader should be marked as exhausted after sampling all data."

    # At this point, the dataloader should be reset. No need to catch StopIteration when calling sample again

def test_dataloader():

    dataset = {
        'inputs': [1, 2, 3, 4, 5],
        'infos': ['a', 'b', 'c', 'd', 'e']
    }
    dataloader = DataLoader(dataset, batch_size=2, randomize=False)

    # Test for-loop usage
    run_for_loop(dataloader)

    # Test next() usage
    run_next(dataloader)

    # Test sample() method
    run_sample(dataloader)

    dataloader._start_new_epoch()  # Manually start a new epoch to reset
    print("Manually started a new epoch.")

    # Test for-loop usage
    run_for_loop(dataloader)

    # Test next() usage
    run_next(dataloader)

    # Test sample() method
    run_sample(dataloader)
