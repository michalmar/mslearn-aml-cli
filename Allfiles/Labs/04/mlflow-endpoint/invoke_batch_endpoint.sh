ENDPOINT_NAME=mlflow-endpoint-diabetes-mma3
DATASET_NAME=diabetes-data-tabular
DATASET_VERSION=1

SUBSCRIPTION_ID=$(az account show --query id | tr -d '\r"')
echo "SUBSCRIPTION_ID: $SUBSCRIPTION_ID"

RESOURCE_GROUP=$(az group show --query name | tr -d '\r"')
echo "RESOURCE_GROUP: $RESOURCE_GROUP"

WORKSPACE=$(az configure -l | jq -r '.[] | select(.name=="workspace") | .value')
echo "WORKSPACE: $WORKSPACE"

SCORING_URI=$(az ml batch-endpoint show --name $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "SCORING_URI: $SCORING_URI"

SCORING_TOKEN=$(az account get-access-token --resource https://ml.azure.com --query accessToken -o tsv)
echo "SCORING_TOKEN: $SCORING_TOKEN"

curl --location --request POST $SCORING_URI \
--header "Authorization: Bearer $SCORING_TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"dataset\": {
            \"dataInputType\": \"DatasetVersion\",
            \"datasetName\": \"$DATASET_NAME\",
            \"datasetVersion\": \"$DATASET_VERSION\"
        },
        \"outputDataset\": {
            \"datastoreId\": \"/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores/workspaceblobstore\",
            \"path\": \"$ENDPOINT_NAME\"
        },
    }
}"


# response=$(curl --location --request POST $SCORING_URI \
# --header "Authorization: Bearer $SCORING_TOKEN" \
# --header "Content-Type: application/json" \
# --data-raw "{
#     \"properties\": {
#         \"InputData\": {
#             \"mnistInput\": {
#                 \"JobInputType\" : \"UriFolder\",
#                 \"Uri\": \"azureml://data/$DATASET_NAME/versions/$DATASET_VERSION/\"
#             }
#         }
#     }
# }")
# echo $response

# JOB_ID=$(echo $response | jq -r '.id')
# JOB_ID_SUFFIX=$(echo ${JOB_ID##/*/})


az ml batch-endpoint invoke --name $ENDPOINT_NAME --input azureml:$DATASET_NAME:$DATASET_VERSION

