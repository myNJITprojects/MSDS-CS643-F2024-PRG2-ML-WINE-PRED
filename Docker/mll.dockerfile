FROM fedora:latest

WORKDIR /home/app

RUN dnf update -y && \
    dnf install python3 poetry java-11-openjdk -y
