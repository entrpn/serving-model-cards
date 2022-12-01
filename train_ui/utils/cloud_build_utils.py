from google.cloud.devtools import cloudbuild_v1

def create_build(project_id,metadata_gcs_uri):
    client = cloudbuild_v1.services.cloud_build.CloudBuildClient()
    build = cloudbuild_v1.Build()

    image_name = f'gcr.io/{project_id}/stable-diffusion-batch-job:latest'

    build.steps = [
        {
            'name' : 'gcr.io/cloud-builders/docker',
            'id' : 'Pull image if exists to use its cache',
            'entrypoint' : 'bash',
            'args' : ['-c', f'docker pull {image_name} || exit 0']
        },
        {
            'name' : 'google/cloud-sdk:alpine',
            'id' : 'Clone repo',
            'entrypoint' : 'git',
            'args' : ['clone', 'https://github.com/entrpn/serving-model-cards.git']
        },
        {
            'name' : 'gcr.io/cloud-builders/gsutil',
            'id' : 'Copy metadata.jsonl',
            'args' : ['cp',metadata_gcs_uri,'serving-model-cards/stable-diffusion-batch-job/metadata.jsonl']
        },
        {
            'name' : 'gcr.io/cloud-builders/docker',
            'id' : 'Build image',
            'args' : [
                'build','-t',image_name,
                '-f','serving-model-cards/stable-diffusion-batch-job/Dockerfile',
                '--cache-from', image_name,
                '.']
        }
    ]

    build.images = [image_name]

    operation = client.create_build(project_id=project_id, build=build)

    return operation.metadata.build.id, 

def get_build(project_id, job_id):
    # Create a client
    client = cloudbuild_v1.CloudBuildClient()

    # Initialize request argument(s)
    request = cloudbuild_v1.GetBuildRequest(
        project_id=project_id,
        id=job_id,
    )

    # Make the request
    response = client.get_build(request=request)
    print(response)
    return response

def get_status_str(status):
    retval = 'STATUS_UNKNOWN'
    if status == 0:
        retval = 'STATUS_UNKNOWN'
    elif status == 1:
        retval = 'QUEUED'
    elif status == 2:
        retval = 'WORKING'
    elif status == 3:
        retval = 'SUCCESS'
    elif status == 4:
        retval = 'FAILURE'
    elif status == 5:
        retval = 'INTERNAL_ERROR'
    elif status == 6:
        retval = 'TIMEOUT'
    elif status == 7:
        retval = 'CANCELLED'
    elif status == 9:
        retval = 'EXPIRED'
    
    return retval
    # STATUS_UNKNOWN = 0;

    # // Build or step is queued; work has not yet begun.
    # QUEUED = 1;

    # // Build or step is being executed.
    # WORKING = 2;

    # // Build or step finished successfully.
    # SUCCESS = 3;

    # // Build or step failed to complete successfully.
    # FAILURE = 4;

    # // Build or step failed due to an internal cause.
    # INTERNAL_ERROR = 5;

    # // Build or step took longer than was allowed.
    # TIMEOUT = 6;

    # // Build or step was canceled by a user.
    # CANCELLED = 7;

    # // Build was enqueued for longer than the value of `queue_ttl`.
    # EXPIRED = 9;