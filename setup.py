from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name = "Medical RAG Chatbot",
    version = "2.1.0",
    author = "Sriram Degala",
    packages = find_packages(),
    install_requires = requirements,
)