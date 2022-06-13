ENDPOINT_NAME=mlflow-endpoint-diabetes-mma3
DATASET_NAME=diabetes_data_file
DATASET_VERSION=latest

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

az ml data create --name diabetes_data_v2 --version 4 --path ./Allfiles/Labs/04/mlflow-endpoint/data/diabetes_classification.csv
az ml data create --name diabetes_data_file --path ./Allfiles/Labs/04/mlflow-endpoint/data/diabetes_classification_8cols.csv --type uri_file 
az ml data create --name diabetes_data_mltable  --path ./Allfiles/Labs/04/mlflow-endpoint/data/ --type mltable

az ml batch-endpoint invoke --name $ENDPOINT_NAME --input azureml:$DATASET_NAME:2

