FROM nvcr.io/nvidia/pytorch:20.08-py3

# Make ssh dir
RUN mkdir /root/.ssh/

# Copying GitHub Deployment private key for the gabrielspmoreira/recsys repo
ADD recsys_repo_key /root/.ssh/recsys_repo_key

# Configures the SSH key
RUN chmod 600 /root/.ssh/recsys_repo_key && \
  echo "IdentityFile /root/ssh/recsys_repo_key" >> /etc/ssh/ssh_config && \
  echo -e "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

# Clone repo
RUN git clone git@github.com:gabrielspmoreira/recsys.git