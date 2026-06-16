"""
Test RC Actor - Standalone test for online reasoning/summarization rollouts

This script:
1. Loads configuration from a Hydra config file (default: conf/test_rc.yaml)
2. Starts vLLM inference servers using the same logic as launch.py
3. Runs RC actor with online rollouts
4. Shows you where to find the output stream

Usage:
    python -m pipelinerl.test_rc_actor
    # or with a different config:
    python -m pipelinerl.test_rc_actor --config-name=your_config
"""
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _popen(
    cmd: list[str],
    env: dict | None = None,
    stdout=None,
    stderr=None,
) -> subprocess.Popen:
    """Wrapper around subprocess.Popen (same as launch.py)"""
    return subprocess.Popen(
        cmd,
        env=env if env else os.environ,
        stdout=stdout,
        stderr=stderr,
        preexec_fn=os.setsid,  # Create new process group
    )


def save_command(log_dir: Path, cmd: list[str]):
    """Save command to file for debugging"""
    with open(log_dir / "command.txt", "w") as f:
        f.write(" ".join(cmd))


def start_actor_llm(
    cfg: DictConfig,
    actor_llm_idx: int,
    local_idx: int,
    gpus: list[int],
    exp_dir: Path
):
    """Start an actor LLM server (same logic as launch.py)"""
    
    finetune_model_path = exp_dir / "finetune" / "current"
    if os.path.exists(finetune_model_path):
        actor_model_path = finetune_model_path
    else:
        actor_model_path = cfg.model_path
        if actor_model_path is None:
            raise ValueError("model_path must be defined")

    log_dir = exp_dir / f"actor_vllm_{actor_llm_idx}"
    os.makedirs(log_dir, exist_ok=True)
    
    # For testing, we use simple OpenAI API server (no weight updates needed)
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(actor_model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(8080 + local_idx),
        "--seed",
        str(cfg.seed + actor_llm_idx),
    ]

    # Add vLLM kwargs as separate arguments
    # Use rc_actor_vllm_config if available, otherwise fall back to vllm_config
    vllm_config = cfg.get('rc_actor_vllm_config') if cfg.get('rc_actor_vllm_config') else cfg.get('vllm_config')
    if vllm_config and hasattr(vllm_config, 'vllm_kwargs') and vllm_config.vllm_kwargs:
        for k, v in vllm_config.vllm_kwargs.items():
            cmd.append(f"--{k}")
            if v not in [None, ""]:
                cmd.append(str(v))

    gpu_str = ",".join([str(gpu) for gpu in gpus])
    logger.info(f"Starting actor_llm {actor_llm_idx} with command: {' '.join(cmd)} on gpus: {gpu_str}")
    save_command(log_dir, cmd)
    
    log_file_path = os.path.join(log_dir, "stdout.log")
    err_file_path = os.path.join(log_dir, "stderr.log")
    
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        process = _popen(
            cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_str},
            stdout=log_file,
            stderr=err_file,
        )
    
    return process


def start_summarization_llm(
    cfg: DictConfig,
    summarization_llm_idx: int,
    local_idx: int,
    gpus: list[int],
    exp_dir: Path
):
    """Start a summarization LLM server (same logic as actor LLM but with different model/config)"""
    
    # Use summarization model if configured, otherwise use actor model
    if cfg.get('summarization_model_path') is not None:
        summarization_model_path = cfg.summarization_model_path
        if summarization_model_path is None:
            raise ValueError("summarization_model_path must be defined")
    else:
        finetune_model_path = exp_dir / "finetune" / "current"
        if os.path.exists(finetune_model_path):
            summarization_model_path = finetune_model_path
        else:
            summarization_model_path = cfg.model_path
            if summarization_model_path is None:
                raise ValueError("model_path must be defined")

    log_dir = exp_dir / f"summarization_vllm_{summarization_llm_idx}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Use summarization-specific port (8280+)
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(summarization_model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(8280 + summarization_llm_idx),
        "--seed",
        str(cfg.seed + 1000 + summarization_llm_idx),  # Different seed space
    ]

    # Add vLLM kwargs - use summarization config if available, otherwise actor config
    vllm_config = cfg.get('summarization_vllm_config') if cfg.get('summarization_vllm_config') else cfg.vllm_config
    if hasattr(vllm_config, 'vllm_kwargs') and vllm_config.vllm_kwargs:
        for k, v in vllm_config.vllm_kwargs.items():
            cmd.append(f"--{k}")
            if v not in [None, ""]:
                cmd.append(str(v))

    gpu_str = ",".join([str(gpu) for gpu in gpus])
    logger.info(f"Starting summarization_llm {summarization_llm_idx} with command: {' '.join(cmd)} on gpus: {gpu_str}")
    save_command(log_dir, cmd)
    
    log_file_path = os.path.join(log_dir, "stdout.log")
    err_file_path = os.path.join(log_dir, "stderr.log")
    
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        process = _popen(
            cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_str},
            stdout=log_file,
            stderr=err_file,
        )
    
    return process


def wait_for_llm_server(port: int, timeout: int = 300) -> bool:
    """Wait for LLM server to be ready"""
    import requests
    
    start_time = time.time()
    url = f"http://localhost:{port}/v1/models"
    
    logger.info(f"Waiting for LLM server on port {port} to be ready...")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"LLM server on port {port} is ready!")
                return True
        except Exception:
            pass
        time.sleep(5)
    
    logger.error(f"LLM server on port {port} did not become ready in {timeout}s")
    return False


def start_environment(cfg: DictConfig, env_idx: int, port: int, exp_dir: Path, job_idx: int):
    """Start an environment server (same logic as launch.py)"""
    
    run_dir = exp_dir / f"environment_{env_idx}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config in the expected location for the environment to read
    config_dir = exp_dir / "conf"
    os.makedirs(config_dir, exist_ok=True)
    
    cmd = [
        "python",
        "-m",
        "pipelinerl.entrypoints.run_environment",
        "--config-dir",
        str(config_dir),
        "--config-name",
        "exp_config",
        f"output_dir={exp_dir}",
        f"hydra.run.dir={str(run_dir)}",
        f"me.job_idx={job_idx}",
    ]
    
    logger.info(f"Starting environment {env_idx} on port {port}")
    save_command(run_dir, cmd)
    
    log_file_path = run_dir / "stdout.log"
    err_file_path = run_dir / "stderr.log"
    
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        process = _popen(
            cmd,
            env=dict(os.environ),
            stdout=log_file,
            stderr=err_file,
        )
    
    return process


def wait_for_environment(port: int, timeout: int = 60) -> bool:
    """Wait for environment server to be ready"""
    import requests
    
    start_time = time.time()
    url = f"http://localhost:{port}/health"
    
    logger.info(f"Waiting for environment server on port {port} to be ready...")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"Environment server on port {port} is ready!")
                return True
        except Exception:
            pass
        time.sleep(3)
    
    logger.error(f"Environment server on port {port} did not become ready in {timeout}s")
    return False


def create_test_dataset() -> List[dict]:
    """Create a small test dataset for reasoning/summarization"""
    return [
        {
            "task": "What is 2 + 2?",
            "answer": "4",
            "dataset": "test_math",
            "id": 0,
        },
        {
            "task": "What is the capital of France?",
            "answer": "Paris",
            "dataset": "test_qa",
            "id": 1,
        },
        {
            "task": "Solve: x + 5 = 10. What is x?",
            "answer": "5",
            "dataset": "test_math",
            "id": 2,
        },
    ]




async def simple_rollout_policy(cfg, llm, problem, session):
    """Simple rollout policy for testing"""
    from pipelinerl.async_llm import llm_async_generate, make_training_text
    from pipelinerl.rollouts import RolloutResult, BaseMetrics
    from tapeagents.core import Prompt
    
    # Create a simple prompt
    messages = [
        {"role": "user", "content": problem["task"]}
    ]
    
    start_time = time.time()
    llm_call = await llm_async_generate(llm, Prompt(messages=messages), session)
    latency = time.time() - start_time
    
    # Create training text
    training_text = make_training_text(llm, llm_call)
    training_text.reward = 0.5  # Dummy reward
    
    # Create metrics
    metrics = BaseMetrics(
        success=False,
        no_error=True,
        no_answer=False,
    )
    
    return RolloutResult(
        training_texts=[training_text],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset", "test"),
    )




def get_actor_urls(num_llms: int, gpus_per_llm: int = 1) -> list[str]:
    """Get actor LLM URLs (same pattern as world_map.get_actor_urls() in launch.py)
    
    Port is determined by the minimum GPU ID allocated to each LLM (local_idx).
    For gpus_per_llm=1: ports are 8080, 8081, 8082, ...
    For gpus_per_llm=2: ports are 8080, 8082, 8084, ... (using GPU 0, 2, 4, ...)
    """
    urls = []
    for i in range(num_llms):
        local_idx = i * gpus_per_llm  # Minimum GPU ID for this LLM
        urls.append(f"http://localhost:{8080 + local_idx}")
    return urls


def prepare_config_for_test(cfg: DictConfig, output_dir: Path, num_llms: int, gpus_per_llm: int = 1, num_summarization_llms: int = 0, summarization_gpus_per_llm: int = 1, num_envs: int = 1, env_start_port: int = 9000, nodelist: list[str] = None, world_size: int = 1):
    """Prepare config for testing. Mainly for setting the llm_urls and job info. Rest everything is set in the config file."""
    
    # Set output directory
    cfg.output_dir = str(output_dir)
    
    if nodelist is None:
        nodelist = ["localhost"]
    
    # Create LLM URLs across all nodes (same pattern as launch.py: line 187, 198)
    llm_urls = []
    llms_per_node = num_llms // world_size
    llms_remainder = num_llms % world_size
    
    global_llm_idx = 0
    for node_rank in range(world_size):
        hostname = nodelist[node_rank]
        # Calculate how many LLMs this node has
        if node_rank < llms_remainder:
            node_num_llms = llms_per_node + 1
        else:
            node_num_llms = llms_per_node
        
        # Generate URLs for LLMs on this node
        for local_llm_idx in range(node_num_llms):
            local_gpu_idx = local_llm_idx * gpus_per_llm
            port = 8080 + local_gpu_idx
            llm_urls.append(f"http://{hostname}:{port}")
            global_llm_idx += 1
    
    # Use OmegaConf to properly set llm_urls (disable struct mode temporarily)
    if not OmegaConf.select(cfg, 'me'):
        OmegaConf.update(cfg, 'me', {})
    
    # Temporarily disable struct mode to add llm_urls
    OmegaConf.set_struct(cfg.me, False)
    cfg.me.llm_urls = "+".join(llm_urls)
    
    # Create separate summarization LLM URLs if configured
    if num_summarization_llms > 0:
        summarization_urls = []
        summarization_llms_per_node = num_summarization_llms // world_size
        summarization_llms_remainder = num_summarization_llms % world_size
        
        global_summ_idx = 0
        for node_rank in range(world_size):
            hostname = nodelist[node_rank]
            # Calculate how many summarization LLMs this node has
            if node_rank < summarization_llms_remainder:
                node_num_summ_llms = summarization_llms_per_node + 1
            else:
                node_num_summ_llms = summarization_llms_per_node
            
            # Calculate base GPU offset on this node (after actor LLMs)
            if node_rank < llms_remainder:
                node_num_actor_llms = llms_per_node + 1
            else:
                node_num_actor_llms = llms_per_node
            base_gpu_offset = node_num_actor_llms * gpus_per_llm
            
            # Generate URLs for summarization LLMs on this node
            for local_summ_idx in range(node_num_summ_llms):
                local_gpu_idx = base_gpu_offset + (local_summ_idx * summarization_gpus_per_llm)
                port = 8280 + global_summ_idx
                summarization_urls.append(f"http://{hostname}:{port}")
                global_summ_idx += 1
        
        cfg.me.summarization_llm_urls = "+".join(summarization_urls)
    
    OmegaConf.set_struct(cfg.me, True)
    
    # Create job information for environments (needed by rollout functions)
    jobs = []
    job_idx = 0
    
    # Add environment jobs
    for env_idx in range(num_envs):
        jobs.append({
            "kind": "environment",
            "idx": job_idx,
            "replica_idx": env_idx,
            "local_idx": 0,
            "node_rank": 0,
            "hostname": "localhost",
            "port": env_start_port + env_idx,
            "gpus": [],
            "url": "",
        })
        job_idx += 1
    
    # Add to config
    OmegaConf.set_struct(cfg, False)
    cfg.jobs = jobs
    OmegaConf.set_struct(cfg, True)
    
    return cfg


@hydra.main(config_path="../conf", config_name="test_rc", version_base="1.3.2")
def main(cfg: DictConfig):
    """Main test function"""
    
    # Calculate number of LLMs and GPU allocation from config (similar to WorldMap)
    import torch
    
    # Multi-node setup: read from environment variables (like WorldMap)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    my_rank = int(os.environ.get("RANK", 0))
    
    # Get node addresses
    address_map = {}
    nodelist = []
    if world_size > 1:
        all_addr = os.environ.get("ALL_ADDR", "")
        if not all_addr:
            raise ValueError("ALL_ADDR environment variable must be set when WORLD_SIZE > 1")
        nodelist = [x.strip() for x in all_addr.strip().split(",")]
        if len(nodelist) != world_size:
            raise ValueError(f"ALL_ADDR length {len(nodelist)} does not match WORLD_SIZE {world_size}")
        master_addr = nodelist[0]
        for rank in range(world_size):
            address_map[rank] = nodelist[rank]
    else:
        master_addr = "localhost"
        address_map[0] = "localhost"
        nodelist = ["localhost"]
    
    logger.info(f"Multi-node setup: world_size={world_size}, my_rank={my_rank}, master_addr={master_addr}")
    logger.info(f"Node list: {nodelist}")
    
    # Get vLLM parallelism config
    # Use rc_actor_vllm_config if available, otherwise fall back to vllm_config
    vllm_config = cfg.get('rc_actor_vllm_config') if cfg.get('rc_actor_vllm_config') else cfg.get('vllm_config')
    if not vllm_config:
        raise ValueError("No vLLM config found! Expected 'rc_actor_vllm_config' or 'vllm_config' in config")
    
    llm_kwargs = vllm_config.vllm_kwargs
    tp = llm_kwargs.get("tensor-parallel-size", 1)
    pp = llm_kwargs.get("pipeline-parallel-size", 1)
    gpus_per_llm = tp * pp
    
    # For testing, we use world config from test_world
    test_world = cfg.get('test_world', {})
    
    # Calculate total GPUs and distribution
    node_size = 8 if world_size > 1 else torch.cuda.device_count()
    total_gpus = world_size * node_size
    
    # Get GPU fractions from config
    rc_actor_fraction = test_world.get('rc_actor_fraction', 0)
    actor_fraction = test_world.get('actor_fraction', 0)
    summarization_fraction = test_world.get('summarization_fraction', 0)
    
    # Calculate GPU allocations
    fraction_sum = rc_actor_fraction + actor_fraction + summarization_fraction
    if fraction_sum == 0:
        # Backward compatibility: if using old config format
        num_llms = test_world.get('actor_fraction', 1)
        num_summarization_llms = test_world.get('summarization_fraction', 0)
    else:
        # New config format: calculate based on fractions
        rc_actor_gpus = int(total_gpus * rc_actor_fraction / fraction_sum) if rc_actor_fraction else 0
        actor_gpus = int(total_gpus * actor_fraction / fraction_sum) if actor_fraction else 0
        summarization_gpus = int(total_gpus * summarization_fraction / fraction_sum) if summarization_fraction else 0
        
        # Calculate number of LLMs
        num_llms = rc_actor_gpus // gpus_per_llm if rc_actor_gpus > 0 else (actor_gpus // gpus_per_llm if actor_gpus > 0 else 1)
        num_summarization_llms = summarization_gpus // gpus_per_llm if summarization_gpus > 0 else 0
    
    # Get summarization vLLM config (if different from actor)
    if cfg.get('summarization_vllm_config') and cfg.summarization_vllm_config.get('vllm_kwargs'):
        summarization_llm_kwargs = cfg.summarization_vllm_config.vllm_kwargs
        summarization_tp = summarization_llm_kwargs.get("tensor-parallel-size", 1)
        summarization_pp = summarization_llm_kwargs.get("pipeline-parallel-size", 1)
        summarization_gpus_per_llm = summarization_tp * summarization_pp
    else:
        summarization_gpus_per_llm = gpus_per_llm
    
    # Distribute LLMs across nodes
    llms_per_node = num_llms // world_size
    llms_remainder = num_llms % world_size
    summarization_llms_per_node = num_summarization_llms // world_size
    summarization_llms_remainder = num_summarization_llms % world_size
    
    # Calculate which LLMs this node should run
    # First node gets extra LLMs if there's a remainder
    if my_rank < llms_remainder:
        my_num_llms = llms_per_node + 1
        my_llm_start_idx = my_rank * (llms_per_node + 1)
    else:
        my_num_llms = llms_per_node
        my_llm_start_idx = llms_remainder * (llms_per_node + 1) + (my_rank - llms_remainder) * llms_per_node
    
    if my_rank < summarization_llms_remainder:
        my_num_summarization_llms = summarization_llms_per_node + 1
        my_summarization_llm_start_idx = my_rank * (summarization_llms_per_node + 1)
    else:
        my_num_summarization_llms = summarization_llms_per_node
        my_summarization_llm_start_idx = summarization_llms_remainder * (summarization_llms_per_node + 1) + (my_rank - summarization_llms_remainder) * summarization_llms_per_node
    
    # Allocate GPUs on this node: each LLM gets gpus_per_llm consecutive GPUs
    gpu_ids = []
    for local_llm_idx in range(my_num_llms):
        llm_gpus = list(range(local_llm_idx * gpus_per_llm, (local_llm_idx + 1) * gpus_per_llm))
        gpu_ids.extend(llm_gpus)
    
    # Allocate GPUs for summarization LLMs on this node (after actor LLMs)
    summarization_gpu_ids = []
    if my_num_summarization_llms > 0:
        base_gpu_offset = my_num_llms * gpus_per_llm
        for local_llm_idx in range(my_num_summarization_llms):
            llm_gpus = list(range(base_gpu_offset + local_llm_idx * summarization_gpus_per_llm, 
                                base_gpu_offset + (local_llm_idx + 1) * summarization_gpus_per_llm))
            summarization_gpu_ids.extend(llm_gpus)
    
    # Create output directory (only rank 0 creates it initially)
    if my_rank == 0:
        output_dir = Path(cfg.output_dir + "_" + str(int(time.time())))
        output_dir.mkdir(parents=True, exist_ok=True)
        # Share output_dir with other ranks via a file
        with open("/tmp/test_rc_actor_output_dir.txt", "w") as f:
            f.write(str(output_dir))
    else:
        # Wait for rank 0 to create and share the directory
        import time as time_module
        for _ in range(30):  # Wait up to 30 seconds
            if os.path.exists("/tmp/test_rc_actor_output_dir.txt"):
                with open("/tmp/test_rc_actor_output_dir.txt", "r") as f:
                    output_dir = Path(f.read().strip())
                break
            time_module.sleep(1)
        else:
            raise RuntimeError("Timeout waiting for output directory from rank 0")
    
    # Environment configuration
    num_envs = test_world.get('env_replicas', 2)
    env_start_port = test_world.get('environment_start_port', 7777)
    actor_group_port = test_world.get('actor_group_port', 9000)
    
    logger.info("=" * 80)
    logger.info(f"RC Actor Test Configuration (Rank {my_rank}/{world_size})")
    logger.info("=" * 80)
    logger.info(f"Model: {cfg.model_path}")
    logger.info(f"GPUs per LLM: {gpus_per_llm}")
    logger.info(f"Total LLM servers (all nodes): {num_llms}")
    logger.info(f"Total summarization LLM servers (all nodes): {num_summarization_llms}")
    logger.info(f"LLM servers on this node: {my_num_llms}")
    logger.info(f"Summarization LLM servers on this node: {my_num_summarization_llms}")
    logger.info(f"LLM index range on this node: {my_llm_start_idx} to {my_llm_start_idx + my_num_llms - 1}")
    logger.info(f"Number of environment servers: {num_envs}")
    logger.info(f"Number of reasoning steps: {cfg.rc_actor.num_reasoning_steps}")
    logger.info(f"Actor GPU allocation (this node): {gpu_ids}")
    if my_num_summarization_llms > 0:
        logger.info(f"Summarization GPU allocation (this node): {summarization_gpu_ids}")
        logger.info(f"Summarization model: {cfg.get('summarization_model_path', cfg.model_path)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    
    # Prepare config (includes job information)
    cfg = prepare_config_for_test(cfg, output_dir, num_llms, gpus_per_llm, num_summarization_llms, summarization_gpus_per_llm, num_envs, env_start_port, nodelist, world_size)
    
    # Save config (environments need to read this)
    config_dir = output_dir / "conf"
    os.makedirs(config_dir, exist_ok=True)
    config_file = config_dir / "exp_config.yaml"
    with open(config_file, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Saved config to {config_file}")
    
    # Start servers and track their ports
    processes = []
    llm_ports = []
    env_ports = []
    
    try:
        # Step 1: Start environment servers (only on rank 0)
        if my_rank == 0:
            logger.info("=" * 80)
            logger.info("Step 1: Starting environment servers")
            logger.info("=" * 80)
            
            for env_idx in range(num_envs):
                port = env_start_port + env_idx
                env_ports.append(port)
                # job_idx for environments (they come first in the jobs list)
                job_idx = env_idx
                logger.info(f"Starting environment {env_idx} on port {port}")
                process = start_environment(cfg, env_idx, port, output_dir, job_idx)
                processes.append(process)
            
            # Wait for environment servers to be ready
            logger.info("\nWaiting for environment servers to be ready...")
            all_env_ready = True
            for port in env_ports:
                if not wait_for_environment(port, timeout=60):
                    all_env_ready = False
                    logger.error(f"Environment server on port {port} failed to start!")
            
            if not all_env_ready:
                logger.error("Not all environment servers started successfully!")
                return 1
            
            logger.info("✅ All environment servers are ready!")
        else:
            logger.info("=" * 80)
            logger.info(f"Step 1: Skipping environment servers (only run on rank 0)")
            logger.info("=" * 80)
        
        # Step 2: Start vLLM servers (only for this node)
        logger.info("=" * 80)
        logger.info(f"Step 2: Starting vLLM servers on node {my_rank}")
        logger.info("=" * 80)
        
        for local_llm_idx in range(my_num_llms):
            # Global LLM index across all nodes
            global_llm_idx = my_llm_start_idx + local_llm_idx
            # Allocate GPUs for this LLM (gpus_per_llm consecutive GPUs on this node)
            llm_gpus = list(range(local_llm_idx * gpus_per_llm, (local_llm_idx + 1) * gpus_per_llm))
            # Use minimum GPU ID as the local_idx (for port assignment)
            local_idx = min(llm_gpus) if llm_gpus else local_llm_idx
            port = 8080 + local_idx
            llm_ports.append(port)
            logger.info(f"Starting LLM {global_llm_idx} (local {local_llm_idx}) on GPUs {llm_gpus}, port {port}")
            process = start_actor_llm(cfg, global_llm_idx, local_idx, llm_gpus, output_dir)
            processes.append(process)
        
        # Wait for all LLM servers to be ready
        logger.info("\nWaiting for all LLM servers to be ready...")
        all_llm_ready = True
        for port in llm_ports:
            if not wait_for_llm_server(port, timeout=300):
                all_llm_ready = False
                logger.error(f"LLM server on port {port} failed to start!")
        
        if not all_llm_ready:
            logger.error("Not all LLM servers started successfully!")
            return 1
        
        logger.info("✅ All vLLM servers are ready!")
        
        # Step 3: Start summarization LLM servers (if configured, only for this node)
        summarization_llm_ports = []
        if my_num_summarization_llms > 0:
            logger.info("=" * 80)
            logger.info(f"Step 3: Starting summarization LLM servers on node {my_rank}")
            logger.info("=" * 80)
            
            base_gpu_offset = my_num_llms * gpus_per_llm
            for local_summ_idx in range(my_num_summarization_llms):
                # Global summarization LLM index across all nodes
                global_summ_idx = my_summarization_llm_start_idx + local_summ_idx
                # Allocate GPUs for this summarization LLM on this node
                llm_gpus = list(range(base_gpu_offset + local_summ_idx * summarization_gpus_per_llm,
                                    base_gpu_offset + (local_summ_idx + 1) * summarization_gpus_per_llm))
                port = 8280 + global_summ_idx
                summarization_llm_ports.append(port)
                logger.info(f"Starting summarization LLM {global_summ_idx} (local {local_summ_idx}) on GPUs {llm_gpus}, port {port}")
                process = start_summarization_llm(cfg, global_summ_idx, local_summ_idx, llm_gpus, output_dir)
                processes.append(process)
            
            # Wait for all summarization LLM servers to be ready
            logger.info("\nWaiting for all summarization LLM servers to be ready...")
            all_summarization_llm_ready = True
            for port in summarization_llm_ports:
                if not wait_for_llm_server(port, timeout=300):
                    all_summarization_llm_ready = False
                    logger.error(f"Summarization LLM server on port {port} failed to start!")
            
            if not all_summarization_llm_ready:
                logger.error("Not all summarization LLM servers started successfully!")
                return 1
            
            logger.info("✅ All summarization LLM servers are ready!")
        
        # Run RC actor (only on rank 0)
        if my_rank == 0:
            logger.info("=" * 80)
            step_num = 4 if my_num_summarization_llms > 0 else 3
            logger.info(f"Step {step_num}: Running RC Actor")
            logger.info("=" * 80)
            
            # Import and run
            from pipelinerl import rc_actor
            
            # Run actor loop (it will set streams backend internally)
            logger.info("Starting RC actor loop...")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Stream file: {output_dir}/streams/actor/0/0/0.jsonl")
            
            rc_actor.run_actor_loop(cfg)
            return 0
        else:
            # Other ranks: keep LLM servers running and wait for termination
            logger.info("=" * 80)
            logger.info(f"Node {my_rank}: LLM servers running, waiting for termination signal...")
            logger.info("=" * 80)
            
            # Wait indefinitely (until interrupted or rank 0 finishes)
            try:
                import signal as sig_module
                sig_module.pause()  # Wait for signal
            except KeyboardInterrupt:
                logger.info(f"Node {my_rank}: Received termination signal")
            return 0
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"\nTest failed with exception: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup: Stop all servers
        logger.info("=" * 80)
        logger.info("Cleanup: Stopping all servers")
        logger.info("=" * 80)
        
        for process in processes:
            if process:
                try:
                    logger.info(f"Stopping process {process.pid}")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=10)
                except Exception as e:
                    logger.warning(f"Error stopping process: {e}")
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except:
                        pass
        
        logger.info(f"\nTest artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
