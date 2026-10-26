# Getting sagemath Docker image
FROM sagemath/sagemath:latest
USER root

# In the docker image sage math executable is at /home/sage/sage/src/bin
# We add the executable path on $PATH
ENV PATH="/home/sage/sage/src/bin:$PATH"

# Get git
RUN apt update && apt upgrade -y
RUN apt install -y git

# Clone Lattice estimator
RUN git clone https://github.com/malb/lattice-estimator.git /lattice-estimator
WORKDIR /lattice-estimator
# Remove param_sweep.py to avoid confusion with paramux's param-sweeper.py
RUN rm param_sweep.py

# Install python requirements
RUN sage -pip install -r /lattice-estimator/requirements.txt
RUN sage -pip install sympy

# Copy source files
COPY param_sweeper.py /lattice-estimator
COPY param_finder.py /lattice-estimator
COPY example_sweeper/ /lattice-estimator
COPY example_finder/ /lattice-estimator

# As the sage command doesn't exist inside the container
# (It the entrypoint of the docker image)
# We add the argument -sh to turn the sage environment
# So we have a shell that can execute sage <py file>
ENTRYPOINT ["sage"]
CMD ["-sh"]
