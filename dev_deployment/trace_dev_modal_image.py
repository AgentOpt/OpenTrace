"""
This sets up a Modal container that runs a Jupyter Lab.
It will:
1. Expose a tunnel for ssh and for jupyter (with public URL link)
2. Add Trace repo code into the container and include it in the PYTHONPATH
3. Attach a persistent volume (for data storage).

The code lives in the ephemeral space, but any results should be saved in the persistent volume,
mounted under `/vol/trace-dev-home/`.
"""

import os
import secrets
import subprocess
from pathlib import Path

import modal

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

# Run this command: `modal volume create trace-dev`
volume = modal.Volume.from_name("trace-dev")

ssh_key_path = Path.home() / ".ssh" / "id_rsa_modal.pub"
project_root = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim()
    .pip_install("jupyter", "graphviz")
    .apt_install("openssh-server")
    .env({
        "PYTHONPATH": "/root/Trace"  # add KernelBench to python path
    })
    .run_commands("mkdir /run/sshd")
    .add_local_file(str(ssh_key_path), "/root/.ssh/authorized_keys", copy=True)
    .add_local_file(str(project_root / "setup.py"), "/root/Trace/setup.py", copy=True)
    .add_local_file(str(project_root / "pyproject.toml"), "/root/Trace/pyproject.toml", copy=True)
    .add_local_file(str(project_root / "README.md"), "/root/Trace/README.md", copy=True)
    .add_local_dir(str(project_root / "opto"), "/root/Trace/opto", copy=True)
    .add_local_dir(str(project_root / "tests"), "/root/Trace/tests", copy=True)
)
app = modal.App("Trace-dev-jupyter", image=image)

@app.function(volumes={"/vol/trace-dev-home/": volume}, timeout=24 * HOURS)
def run_jupyter():

    # set up Trace env
    subprocess.run(["pip", "install", "-e", "/root/Trace"])

    # a nested setup to start SSH connection and Jupyter

    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])

    with modal.forward(port=22, unencrypted=True) as ssh_tunnel:
        hostname, port = ssh_tunnel.tcp_socket
        connection_cmd = f'ssh -p {port} root@{hostname}'
        print("Run `ssh-add ~/.ssh/id_rsa_modal` to add SSH key credential")
        print(f"ssh into container using: {connection_cmd}")

        token = secrets.token_urlsafe(13)

        with modal.forward(8888) as tunnel:
            url = tunnel.url + "/?token=" + token
            print(f"\033[91mStarting Jupyter at {url}\033[0m")
            subprocess.run(
                [
                    "jupyter",
                    "notebook",
                    "--no-browser",
                    "--allow-root",
                    "--ip=0.0.0.0",
                    "--port=8888",
                    "--LabApp.allow_origin='*'",
                    "--LabApp.allow_remote_access=1",
                ],
                env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"},
                stderr=subprocess.DEVNULL,
        )

@app.local_entrypoint()
def main():
    subprocess.run(["ssh-add", "~/.ssh/id_rsa_modal"])
    run_jupyter.remote()
