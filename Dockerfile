# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy Pipfile and Pipfile.lock to the working directory
COPY Pipfile Pipfile.lock /app/

# Install pipenv
RUN pip install pipenv

# Install project dependencies
RUN pipenv install --deploy --ignore-pipfile

# Copy the contents of the test directory into the container at /app
COPY test/ /app
COPY src/model.py /app/

# Copy the data into the container at /app
COPY data/news/topic_doc_mean_n5000_k3477_seed_1.csv.x /app/data/

# Copy the model into the container at /app
COPY models/news/0/checkpoint.pth /app/models/

# Expose port 80
EXPOSE 80

# Run app.py when the container launches
CMD ["pipenv", "run", "python", "app.py"]
