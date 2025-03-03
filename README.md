# Homework2 BHARGAV SAI ALLAM 700752883
Cloud Computing & CNN Tasks
Question 1: Cloud Computing for Deep Learning
Elasticity and Scalability in Cloud Computing

Elasticity dynamically adjusts computing resources based on workload demands, optimizing cost and resource usage for deep learning tasks.
Scalability enables a system to manage increasing workloads by either upgrading existing resources (vertical scaling) or adding more instances (horizontal scaling), ensuring efficient model training on larger datasets.
Comparison of AWS SageMaker, Google Vertex AI, and Microsoft Azure Machine Learning Studio

AWS SageMaker supports TensorFlow, PyTorch, and MXNet, offering SageMaker Autopilot for AutoML and distributed training with managed spot instances. It provides one-click deployment, pay-as-you-go pricing, and seamless integration with AWS services like S3 and Lambda.

Google Vertex AI supports TensorFlow, PyTorch, and JAX, featuring Vertex AI AutoML and distributed training with GPU/TPU acceleration. It offers an end-to-end pipeline, flexible pricing, and integration with Google Cloud services such as BigQuery.

Microsoft Azure Machine Learning Studio supports TensorFlow, PyTorch, and Scikit-learn, providing Azure AutoML, distributed training with GPU/CPU clusters, and real-time and batch deployment. It includes reserved instance pricing options and integrates with Azure services like Azure Storage and Synapse.

Question 2: Convolution Operations with Different Parameters
Using NumPy and TensorFlow/Keras, a 5×5 input matrix is convolved with a 3×3 kernel under different configurations:

Stride = 1, Padding = 'VALID' results in an output feature map with reduced dimensions.
Stride = 1, Padding = 'SAME' maintains the original input size.
Stride = 2, Padding = 'VALID' significantly reduces spatial dimensions.
Stride = 2, Padding = 'SAME' preserves structural integrity while reducing size.
Question 3: CNN Feature Extraction
Edge Detection using Sobel Filters

Load a grayscale image.
Apply Sobel-X and Sobel-Y filters to highlight edges along horizontal and vertical directions.
Display the original image alongside the filtered results.
Max Pooling and Average Pooling

Generate a random 4×4 matrix to simulate an input image.
Apply 2×2 Max Pooling to extract dominant features.
Apply 2×2 Average Pooling to smooth the feature map.
Compare the original and pooled matrices to analyze feature retention.
Question 4: Implementing and Comparing CNN Architectures
Implementing AlexNet

Construct AlexNet using TensorFlow/Keras with:
Conv2D layers with varied kernel sizes and activation functions.
MaxPooling layers for spatial dimension reduction.
Fully connected layers using ReLU activations.
Dropout layers to prevent overfitting.
Print the model summary to review its structure.
Comparing AlexNet, VGG, and ResNet
AlexNet consists of 8 layers with large filters (11×11, 5×5), following a simple sequential architecture that delivers solid performance.

VGG features 16-19 layers with small (3×3) filters, adopting a deep but uniform filter approach that improves accuracy.

ResNet includes 50+ layers with residual blocks and skip connections, making it the most effective for deep learning models by addressing vanishing gradient issues.
