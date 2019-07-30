# aws-neo
aiSage support aws sagemaker neo runtime

## How to install sagemaker neo runtime on aiSage
- run the command "sh sagemakerNeo.sh" in terminal

## Run the demo program
- step1
  ##### copy demo-mxnet-ssd-mobilenet-512-dlr-thread.py to neo-ai-dlr/demo/aisage/ folder
- step2 dowload pre-build model 
  ##### change to neo-ai-dlr/demo/aisage/models
  ##### mkdir mxnet-ssd-mobilenet-512
  ##### cd mxnet-ssd-mobilenet-512
  ##### curl -O https://s3.us-east-2.amazonaws.com/dlc-models/aisage/mxnet-ssd-mobilenet-512/model.params
  ##### curl -O https://s3.us-east-2.amazonaws.com/dlc-models/aisage/mxnet-ssd-mobilenet-512/model.json
  ##### curl -O https://s3.us-east-2.amazonaws.com/dlc-models/aisage/mxnet-ssd-mobilenet-512/model.so

- step3 run the sample code
  ##### cd neo-ai-dlr/demo/aisage/
  ##### python3 demo-mxnet-ssd-mobilenet-512-dlr-thread.py
