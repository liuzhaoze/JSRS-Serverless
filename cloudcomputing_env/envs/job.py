class Job:
    def __init__(self, job_id: int, arrival_time: float, length: float, parallelism: int, region: int, data_size: int):
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.length = length
        self.parallelism = parallelism
        self.region = region
        self.data_size = data_size

    def __repr__(self) -> str:
        return f"Job(id={self.job_id}, arrival_time={self.arrival_time:.2f}, length={self.length:.2f}, parallelism={self.parallelism}, region={self.region}, data_size={self.data_size})"
