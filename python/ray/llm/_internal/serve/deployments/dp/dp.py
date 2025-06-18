
from ray import serve


class VLLMDPEngine:
    
    def __init__(self):
        
        parallel_config = vllm_config.parallel_config
        
        dp_size = parallel_config.data_parallel_size
        local_engine_count = parallel_config.data_parallel_size_local
        host = parallel_config.data_parallel_master_ip
        local_only = local_engine_count == dp_size

        # Set up input and output addresses.
        input_addresses = [
            get_engine_client_zmq_addr(local_only, host)
            for _ in range(num_api_servers)
        ]
        output_addresses = [
            get_engine_client_zmq_addr(local_only, host)
            for _ in range(num_api_servers)
        ]

        addresses = EngineZmqAddresses(
            inputs=input_addresses,
            outputs=output_addresses,
        )

        # Set up coordinator for dp > 1.
        coordinator = None
        stats_update_address = None
        if dp_size > 1:
            coordinator = DPCoordinator(parallel_config)
            addresses.coordinator_input, addresses.coordinator_output = (
                coordinator.get_engine_socket_addresses())
            stats_update_address = coordinator.get_stats_publish_address()
            logger.info("Started DP Coordinator process (PID: %d)",
                        coordinator.proc.pid)
        
        
        engine_actor_manager = CoreEngineActorManager(
            local_engine_count=local_engine_count,
            vllm_config=vllm_config,
            addresses=addresses,
            executor_class=Executor.get_class(vllm_config),
            log_stats=not engine_args.disable_log_stats,
        )

