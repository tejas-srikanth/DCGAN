import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

ROLE = os.environ.get("ROLE")
S3_BUCKET = os.environ.get("S3_BUCKET")

sagemaker_session = sagemaker.Session(boto3.session.Session(region_name='us-east-2'))

pytorch_estimator = PyTorch(entry_point='train.py',
                            role=ROLE,
                            framework_version='1.8.0',
                            py_version='py3',
                            instance_count=1,
                            region_name='us-east-2',
                            sagemaker_session=sagemaker_session,
                            instance_type='ml.p2.xlarge',
                            hyperparameters={
                                'epochs': 5,
                            })

pytorch_estimator.fit(S3_BUCKET)