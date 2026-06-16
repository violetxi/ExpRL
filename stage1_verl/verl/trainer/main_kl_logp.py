"""
Entrypoint for KL analysis logprob computation using verl's distributed infrastructure.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_kl_logp import RayKLLogp
from verl.utils.device import is_cuda_available


@hydra.main(config_path="config", config_name="logp", version_base=None)
def main(config):
    run_kl_logp(config)


def run_kl_logp(config) -> None:
    if not ray.is_initialized():
        runtime_env = get_ppo_ray_runtime_env()
        from verl.utils.fs import copy_to_local
        local_env_path = copy_to_local(config.trainer.get("env", ".env"))
        from dotenv import load_dotenv
        load_dotenv(local_env_path, override=True)
        if "HF_TOKEN" in os.environ:
            runtime_env["env_vars"]["HF_TOKEN"] = os.environ["HF_TOKEN"]

        detected_ip = socket.gethostbyname(socket.gethostname())
        os.environ["RAY_NODE_IP_ADDRESS"] = detected_ip

        ray.init(
            runtime_env=runtime_env,
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        import os
        import socket
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Only need ActorRollout, no critic/reward model
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # KL-specific configs
        data_path = config.data.get("kl_data_path", None)
        output_path = os.path.join(config.trainer.default_local_dir, "logprobs.pkl")
        solution_format = config.data.get("solution_format", "assistant_prefix")
        batch_size = config.data.get("kl_batch_size", 32)

        trainer = RayKLLogp(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            data_path=data_path,
            output_path=output_path,
            solution_format=solution_format,
            batch_size=batch_size,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
