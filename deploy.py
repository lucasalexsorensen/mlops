from google.cloud import aiplatform


CUSTOM_PREDICTOR_IMAGE_URI = (
    "europe-west4-docker.pkg.dev/dtumlops-project-mask-no-mask/dtu-mlops/serve"
)
VERSION = 1
model_display_name = f"mask-v{VERSION}"
model_description = "PyTorch based text classifier with custom container"
MODEL_NAME = "mask"
health_route = "/ping"
predict_route = f"/predictions/{MODEL_NAME}"
serving_container_ports = [7080]


model = aiplatform.Model.upload(
    display_name=model_display_name,
    description=model_description,
    serving_container_image_uri=CUSTOM_PREDICTOR_IMAGE_URI,
    serving_container_predict_route=predict_route,
    serving_container_health_route=health_route,
    serving_container_ports=serving_container_ports,
)
model.wait()
print(model.display_name)
print(model.resource_name)

endpoint_display_name = f"mask-endpoint"
endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)

traffic_percentage = 100
machine_type = "n1-standard-4"
deployed_model_display_name = model_display_name
min_replica_count = 1
max_replica_count = 3
sync = True

model.deploy(
    endpoint=endpoint,
    deployed_model_display_name=deployed_model_display_name,
    machine_type=machine_type,
    traffic_percentage=traffic_percentage,
    sync=sync,
)

model.wait()
print("DEPLOYED TO ENDPOINT!")
