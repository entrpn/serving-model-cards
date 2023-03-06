from google.cloud import dataflow_v1beta3


def sample_create_job():
    # Create a client
    client = dataflow_v1beta3.JobsV1Beta3Client()

    # Initialize request argument(s)
    request = dataflow_v1beta3.CreateJobRequest(
    )

    # Make the request
    response = client.create_job(request=request)

    # Handle the response
    print(response)