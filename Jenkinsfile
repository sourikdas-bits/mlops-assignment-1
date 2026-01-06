pipeline {
    agent any

    environment {
        // Example: PATH = "/usr/local/bin:${env.PATH}"
    }

    stages {
        stage('Clone Repository') {
            steps {
                withCredentials([string(credentialsId: 'GITHUB_PAT', variable: 'GITHUB_TOKEN')]) {
                    git url: "https://sourikdas-bits:${GITHUB_TOKEN}@github.com/sourikdas-bits/mlops-assignment-1.git", branch: 'assignment'
                }
            }
        }

        stage('Linting') {
            steps {
                sh 'pip install flake8'
                sh 'flake8 .'
            }
        }

        stage('Unit Testing') {
            steps {
                sh 'pip install pytest'
                sh 'pytest tests/'
            }
        }

        stage('Model Training') {
            steps {
                sh 'python source/2_3_4_build_model.py'
            }
        }
    }

    post {
        always {
            echo 'Pipeline completed.'
        }
    }
}
