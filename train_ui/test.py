from utils.cloud_build_utils import create_build
from utils.vertexai_utils import get_custom_job_sample

# job = get_custom_job_sample('jfacevedo-demos','573945207138025472','us-central1')
# print(job.state)
# print(type(job.state.value))
def odd(max):
    n = 0
    while n < max:
        yield n
        n = n + 1
    yield 'done'
    return


for x in odd(3):
    print(x)