# docker build -t registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/3dprinting-docker-gpu:$1 -f docker/docker_3dprinting_gpu/Dockerfile .
docker build -t registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/3dprinting-docker:$1 -f docker/docker_3dprinting/Dockerfile .

docker push registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/3dprinting-docker:$1
# docker push registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/3dprinting-docker-gpu:$1