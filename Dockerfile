FROM python:3.10-slim-bullseye

# 创建项目目录并切换到该目录
#RUN mkdir /soft
WORKDIR /soft

# 复制当前目录下的所有内容到镜像中的 /soft 目录
COPY . /soft

# 赋予 setup_environment.sh 可执行权限
RUN chmod +x ./setup_environment.sh

# 执行 setup_environment.sh 并确保遇到错误立即退出
RUN set -e && ./setup_environment.sh

# 声明容器将使用的端口
EXPOSE 5130

# 运行应用程序
CMD ["python", "app.py"]
