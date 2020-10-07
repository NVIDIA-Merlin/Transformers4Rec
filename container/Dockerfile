FROM nvcr.io/nvidia/pytorch:20.08-py3

################ Setup GitHub repo ################

# Make ssh dir
RUN echo "Setting up GitHub repo"; mkdir /root/.ssh/

# Copying GitHub Deployment private key for the gabrielspmoreira/recsys repo
ADD recsys_repo_key /root/.ssh/recsys_repo_key

# Configures the SSH key
RUN chmod 600 /root/.ssh/recsys_repo_key && \
  echo "IdentityFile /root/.ssh/recsys_repo_key" >> /etc/ssh/ssh_config && \
  echo -e "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

WORKDIR /workspace

# Clone repo
RUN git clone git@github.com:gabrielspmoreira/recsys.git

#Checkout experimentation branch
WORKDIR recsys
RUN git checkout -b experimentation origin/experimentation

################ Setup GitHub repo ################
WORKDIR /workspace/recsys/transformers4recsys/codes

#echo "Installing dependencies"
RUN pip install -r requirements.txt;