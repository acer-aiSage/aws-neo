# Sagemaker neo for aiSage
aiSage support aws sagemaker neo runtime. You can install the sagemaker neo runtime to run the neo models on aiSage

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
  
## Sagemaker Neo Compiler 
#### Upload your model to S3
#### If needed install and configure the AWS CLI
#### Invoke Neo with “RK3399” as the target from the AWS CLI
      - a.	Use the following command: aws sagemaker create-compilation-job                                                --cli-input-json file:///tmp/job.json --region us-west-2
      - b.	See Appendix A for a sample json you can provide
##### Download model from S3
##### Deploy to device and run with DLR.
