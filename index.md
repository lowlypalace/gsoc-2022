# GSoC 2022 | Python Software Foundation (Activeloop)
---

This is a summary of the work I did for Activeloop's open-source open source package named [Hub](https://github.com/activeloopai/Hub) under the Python Software Foundation organization as part of Google Summer of Code 2022.

## Project Details
- Title: [Automated Dataset Tuning](https://summerofcode.withgoogle.com/programs/2022/projects/o6DWVntH)
- Organization: [Python Software Foundation](https://www.python.org/psf/)
- Repository: [Activeloop Hub](https://github.com/activeloopai/Hub)
- Mentors: [Davit Buniatyan](https://github.com/davidbuniat), [Fariz Rahman](https://github.com/farizrahman4u), [Ivo Stranic](https://github.com/istranic), [Mikayel Harutyunyan](https://github.com/mikayelh)

---

## Contributions
- Pull Request: [Cleanlab + Skorch Integration](https://github.com/activeloopai/Hub/pull/1821)
- Tutorial of the Workflow (Colab): [Finding Label Errors in Image Classification Dataset](https://colab.research.google.com/drive/1ufji2akWX0r6DcUD70vK3KiBvq0m6xbq?usp=sharing)
- Blog Post: TBA

---

## Introduction

This summer, I got accepted as a contributor to the Python Software Foundation (Activeloop) in the 2022 Google Summer of Code. Activeloop's open-source open-source package named [Hub](https://github.com/activeloopai/Hub) lets you store (even petabyte-scale) datasets on the cloud and lets you seamlessly integrate them with your ML applications. The goal is to develop a set of data-centric APIs for a machine learning library that can optimize datasets with minimal parameters. I had about a month to come up with a technical solution. In this one month, I had to synthesize the requirements, research a large number of data-centric strategies (e.g., data augmentation, noise cleaning, data selection, self-supervision), review academic papers, and develop end-to-end pipelines for machine learning experiments to benchmark the performance of data-centric strategies for various computer vision tasks (e.g., classification). Taking into account all the possible strategies, there exists a huge number of options. Each of these specific strategies has hundreds of hyperparameters, and the way they are structured impacts the downstream model.

This is an ambiguous task as you need to be capable of understanding the high-level business problem down to the lines of code. If I were a user and this hadn't been implemented yet, how would I go about doing it myself? Asking this question helped to understand the end-user and uncover that the process is highly iterative. I communicated with mentors, built a high-level overview of the process, then broke it down into subproblems and separately optimized each component. This has helped me to focus on exploring and evaluating each strategy one at a time without losing attention to the ambiguous high-level problem I'm trying to solve. As the project has a time constraint, I had to challenge myself with a question: *"How can I implement a strategy that will yield the most impact for the end-users, given the tight timeline?"*

## Research Phase
Early on during the research phase, the mentors challenged me on what could be done before proceeding with some advanced strategies, such as data augmentation or data selection. Are there any fundamental flaws in the data we have at hand? What could be done in a subset of ML problems, such as supervised learning? Today, most practical machine learning models utilize supervised learning. For supervised learning to work, you need a labeled set of data that the model can learn from to make correct decisions. Data labeling typically starts by asking humans to make judgments about a given piece of unlabeled data. For example, labelers may be asked to tag all the images in a dataset where an image contains a *car*. The machine learning model uses human-provided labels to learn the underlying patterns in a process called *model training*. The result is a trained model that can be used to make predictions on new data. Supervised learning is the [dominant ML system at Google](https://developers.google.com/machine-learning/intro-to-ml/supervised). Because supervised learning's tasks are well-defined, like identifying a class of an image, it has more potential use cases than unsupervised learning. 

In machine learning, a properly labeled dataset that you use as the objective standard to train and assess a given model is often called *ground truth*. The accuracy of your trained model will depend on the accuracy of your ground truth, so spending the time and resources to ensure highly accurate data labeling is essential. 
If you've ever used datasets like CIFAR, MNIST, ImageNet, or IMDB, you likely assumed the class labels are correct. Supervised ML often assumes that the labels we train our model on are correct, but [recent studies](https://www.technologyreview.com/2021/04/01/1021619/ai-data-errors-warp-machine-learning-progress/) have discovered that even highly-curated ML benchmark datasets are full of [label errors](https://labelerrors.com/). What's more, the [Northcutt's NeurIPS 2021](https://arxiv.org/abs/2103.14749) work on analyzing errors in datasets found out hundreds of samples across popular datasets where an agreement could not be reached on true ground truth despite looking at collating outcomes from labelers. Furthermore, the labels in datasets from real-world applications can be of [far lower quality](https://go.cloudfactory.com/hubfs/02-Contents/3-Reports/Crowd-vs-Managed-Team-Hivemind-Study.pdf). There are several factors that lead to error in the dataset, such as a human error made while annotating the examples. These days, it is increasingly the training data, not the models or infrastructure, that decides whether machine learning will be a success or failure. However, training our ML models to predict fundamentally flawed labels seems problematic. This becomes especially problematic when these errors reach test sets, the subsets of datasets used to validate the trained model. 
Even worse, we might train and evaluate these models with flawed labels and deploy the resulting models at scale. 

---

## Hub + Cleanlab
Hub community has uploaded a variety of popular machine learning datasets like [CIFAR-10](https://docs.activeloop.ai/datasets/cifar-10-dataset), MNIST or Fashion-MNIST, and [ImageNet](https://docs.activeloop.ai/datasets/imagenet-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme). Without any need to download, these datasets can be accessed and streamed with Hub with one line of code. This enables you to explore the datasets and train models without downloading machine learning datasets regardless of their size. However, most of these datasets contain [label errors](https://labelerrors.com/). What can we do about this? To tackle this problem, the Northcutt's group of researchers co-founded [Cleanlab](https://cleanlab.ai/), a tool that allows to automatically find and fix label errors in ML datasets. Under the hood, it uses [Confident Learning](https://arxiv.org/abs/1911.00068) (CL) algorithm to detect label errors. 

Nevertheless, the Hub dataset is a [specific format of a dataset](https://docs.activeloop.ai/how-hub-works/data-layout) that uses a columnar storage architecture, and the columns are referred to as tensors. It is not trivial how to set up the workflow for finding label errors, as Cleanlab does not support Hub datasets. Therefore, users are required to make their models and data format compatible and then run a training pipeline that would allow them to utilize Cleanlab to find label issues. How can we abstract this complexity and provide users with a clean and intuitive interface to find label errors? Furthermore, how can we use Hub's powerful [data visualization](https://docs.activeloop.ai/dataset-visualization) tools to enable users to visualize these insights and make informed decisions on what to do with noisy data?

---

## Experiments
We've talked about how label errors can be detrimental to the success of an ML project. In theory, this assumption holds a strong argument. However, we wanted to prove it by running a few experiments before introducing this feature to our users. 

By communicating with my mentors, we set up a few research questions that we were looking to answer:

> ***Research Question 1**: What is the impact of label errors on a model performance? Can we implement an experiment that will enable us to quantify the impact? How do we ensure that our experiments are unbiased and reproducible?*

> ***Research Question 2**: Assuming that the label errors could be found, what would be the following steps to fix the errors? Can we find a way to prune or relabel noisy examples from a dataset automatically? Does it make sense to leave some examples but correct their labels?*

### Label Errors Impact on Model Performance 

Let's try to break down the first research question. To measure the impact of label errors, we'll need to find a metric to optimize for. Accuracy is one metric for evaluating supervised models, which is the fraction of predictions that a model got right. Now, to quantify the impact of label errors, we'll need a way to benchmark a model trained on clean data and a model trained on noisy data. By looking at the accuracy of the predictions made by a model trained on noisy data and on clean data, we can compare the approaches against each other. Let's suppose we can compare this metric on a fixed model trained on some dataset. However, we don't necessarily know the ground truth, as there could already be errors in the dataset. We can only be confident if we introduce these label errors ourselves, assuming that we do it on a dataset with a relatively low label error rate in the first place. To overcome this, we can artificially introduce some noise to a dataset that we assume has a low rate of label errors. We can introduce random noise to the training set by randomly flipping the labels. 

For the experiment, let's use [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset and assume that it has a low rate of errors. Then, we gradually introduce **5%** of the noise at each step, comparing the performance of `Baseline` and `Cleanlab` in parallel. In this case, `Baseline` is a model trained on noisy data, while `Cleanlab` is a model trained on clean data. Here's an example: *if we have **60,000** samples in the dataset, at **10%** of the noise, we would randomly flip **6,000** labels*. The `Baseline` would then be trained on all **60,000** samples, but the `Cleanlab` would be trained only on the examples that weren't classified as erroneous. 

![enter image description here](https://github.com/lowlypalace/gsoc-2022/raw/main/accuracy_plot.png)

It looks like pruning the noisy samples works well, as Cleanlab seems to detect accurately the erroneous labels that were introduced at each noise level. We can also see that the accuracy stays around **85%** with Cleanlab across a range of noise levels, while with the random noise (without removing any samples), it drops at a much higher rate to a low of **65%** at the highest noise level.


![enter image description here](https://github.com/lowlypalace/gsoc-2022/raw/main/samples_plot.png)
Looking at the graph above, we can see that Cleanlab prunes more label errors than we introduced. We can argue that there might have been some initial label errors in the dataset before we introduced them ourselves. However, it might be that Cleanlab is overestimating. Nevertheless, it seems to be able to systematically pick up the newly introduced noisy labels and identify them as erroneous. As we mentioned, the foundation CL depends on is that label noise is class-conditional, depending only on the true latent class, not the data. In the noisy real-world data, the mislabelling between different classes would have a stronger statistical dependence, so we can say that this example was even a bit more difficult for Cleanlab as we swapped the labels randomly.

### Fixing Label Errors
Next, let's try to break down the second research question. Now that we have discovered that we can systematically detect label errors, what would be the next steps? As a user, I want to be able to get insights and do something to my noisy data. If a sample is mislabeled but has meaningful data, do I leave out these examples and relabel them to the correct labels? Do I remove this example if a sample is simply noise (e.g., a corrupted image)? Furthermore, is there an automatic way to set this up?

Let's try to break it down further. Again, we'll need a way to benchmark the approaches and compare them against each other. What if we ask ourselves a simpler question: *now that we have label errors, what if we remove the examples that are the noisiest (i.e., have little or no meaningful data) and attempt to fix or relabel the less noisy examples.* As Cleanlab provides us with quality scores for each example, we can try to prune the labels with the lowest quality scores and relabel the rest. Assumably, after training a model on clean data, we should be able to get the updated predictions for each label, which we can use to relabel the less noisy labels. 

As a next step, we try to set a threshold that would tell us the ratio of examples we will remove to the examples we will relabel. For example, with a threshold of **10%**, we will prune the first top **10%** examples with the lowest label quality and relabel the rest of the examples. We can then see how the accuracy across various noise levels compares. Here, we tried to experiment with different threshold values for pruning and relabelling the images. We started with a threshold of **0%** (e.g., relabel all labels to the guessed labels) and then gradually increased the threshold value with a **10%** step until we reached **100%** prune level (remove all samples found to be erroneous). 

![enter image description here](https://github.com/lowlypalace/gsoc-2022/raw/main/accuracy_threshold_plot.png)

On the graph, we plotted the accuracy of the models trained with training sets that were fixed with different threshold values. For example, `100% Prune / 0% Relabel` indicates the accuracy of the model when all erroneous samples and their labels were deleted, while `0% Prune / 100% Relabel` shows the accuracy of the model when all of the samples were left but relabelled.

Looking at the graph, we can say that Cleanlab does a great job identifying labels but not necessarily relabeling them automatically. As soon as we increase the number of labels we'd like to relabel, the accuracy starts to go down linearly. The training set with **100%** of pruning got the highest accuracy, while the training set with all labels relabelled got the worst accuracy. The confident learning approach can sometimes get it wrong, too, like confusing a correctly labeled image. Therefore, it is best to go through the examples in a dataset and decide whether to remove or relabel an example. With these insights, we know that it makes sense to provide users with the functionality to prune erroneous examples automatically. However, we might want to give the users some decision-making tools, such as quickly visualizing the labels that are most likely to be erroneous.

---

## Development Phase
After running the experiments, we now have enough insights that should allow us to move the next step. We'll need to provide an easy-to-use and clean interface to find label errors automatically and use these insights to make informed decisions on what to do next with these errors.

Hub hosts a variety of datasets such as audio, image, object detection, and text datasets. However, Hub's focus is primarily on computer vision datasets. Within this domain, image classification, a task of assigning a label or class to an entire image,  is one of the most common ML problems. Images are expected to have only one class for each image. Image classification models take an image as input and return a prediction about which class the image belongs to. To scope out the problem, the support for image classification tasks was one of the first goals. In the first iteration, I created the first version of the API. By running the experiments, I had a good grasp on the internals of Cleanlab. After two weeks of coding, it was the time to finally put it all together in the first draft PR. Over the next few weeks, I had a number of syncs with the team where we iterated on the solution. 

### Skorch Integration
As some  `cleanlab`  features leverage  `scikit-learn`  compatibility, I had to deploy a wrapper for deep learning frameworks, such as PyTorch and Tensorflow to make them compatible. As PyTorch has been widely used within Hub's community, I implemented a wrapper for this library first. I wrapped the the neural net using  [`skorch`](https://skorch.readthedocs.io/en/stable/), which makes any PyTorch module scikit-learn compatible. Specifically, `skorch` provides a class  [`NeuralNet`](https://skorch.readthedocs.io/en/stable/net.html#skorch.net.NeuralNet "skorch.net.NeuralNet") that wraps the PyTorch [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "(in PyTorch v1.10.1)") while providing an interface that should be familiar for sklearn users. However, as we mentioned before, Hub stores datasets in columnar format, where columns are reffered as *tensors*. For example, a simple image classification dataset might contain two tensors *images* and *labels*. To access an example of a `dataset`, we use `dataset.images[0]` to get the first image of a dataset and `dataset.labels[0]` to get the first label of a dataset. Therefore, I had to create a class that extends the `NeuralNet` class from `skorch` and overrides a few methods to be able to fetch examples in the training and validation loops. Another advantage of using `skorch` is that it also abstracts away the training loop and makes us write less boilerplate code to find label issues in a dataset. 

To make our model `scikit-learn` compatible, we'll use  `skorch()`  method from the `integrations` module. This function will wrap a PyTorch Module in a skorch  `NeuralNet`  and allow us to get access to common scikit-learn methods such as  `predict()`  and  `fit()`.

```python
from hub.integrations import skorch

model = skorch(
    dataset=ds,
    epochs=15,  # Set the number of epochs.
    batch_size=8,
    transform=transform,
    shuffle=True, # Shuffle the data before each epoch.
) 
```

Here, we won't be defining a custom PyTorch module, but feel free to pass any PyTorch network as  `module`  parameter to  `skorch()`  method. As we don't specify any module, the predefined model  `torchvision.models.resnet18()`  will be used by default. We've set  `epochs=15`  above, but feel free to lower this value to train the model faster or increase it to improve the results. There are many other useful parameters that can be passed in to a  `skorch`  instance, such as  `valid_dataset`  to be used for the validation. 

### Cleanlab Integration

Let’s look at how we can find label errors in a Hub dataset. We will use a `clean_labels()` method from `cleanlab` module. In general, `cleanlab` uses predictions from a trained classifier to identify label issues in the data. However, if we train a classifier on some of this data, then its predictions on the same data will become untrustworthy due to overfitting. To overcome this, the function will run cross-validation to get **out-of-sample** predicted probabilities for each example in the dataset.

```python
from hub.integrations.cleanlab import clean_labels

label_issues = clean_labels(
    dataset=ds,
    model=model,  # Pass instantiated skorch module.
    folds=5,  # Set the number of folds for cross-validation.
)
```

`label_issues` is a pandas DataFrame of label issues for each example. Each row represents an example from the dataset and the DataFrame contains the following columns:

-   `is_label_issue`: A boolean mask for the entire dataset where `True` represents a label issue and `False` represents an example that is confidently/accurately labeled.
-   `label_quality`: Label quality scores (between `0` to `1`) for each datapoint, where lower scores indicate labels less likely to be correct.
-   `predicted_labels`: Class predicted by model trained on cleaned data.

Here's an example of how it might look like:

|  | is_label_issue |label_quality | predicted_labels|
|--|--|--|--|
| 0 | False | 0.423514| 1 |
| 1 | True |0.001363 | 0 |
| 2 | False | 0.998112 | 3 |

To visualize the labels in [Dataset Visualization](https://docs.activeloop.ai/dataset-visualization), we can use a method `create_tensors()` that will help us to automatically create a tensor group `label_issues`. The tensors in this group would represent the columns `is_label_issue`, `label_quality` and `predicted_labels`. 

```python
create_tensors(
    dataset=ds_train,
    label_issues=label_issues,  # Pass label_issues computed before.
    branch="main",  # Commit changes to main branch.
)
```

Now, we can easily sort the labels with the lowest label quality scores by sorting on  `label_issues/label_quality_scores`  tensor. The label quality score is between  `0`  and  `1`, where  `0`  is a dirty label and  `1`  is a clean label. In general, you should first analyze the samples that have the lowest quality scores as these examples are most likely to be incorrect. Therefore, before moving further down the list, you can remove or relabel these samples. Additionally, we can view all labels with errors by running a query:  `select * where "label_issues/is_label_issue" == true'`. We can also take a look at the guessed labels for each example in a dataset by viewing  `label_issues/is_label_issue`  tensor.

Another handy method is `clean_view()`, which allows us to get a view of the dataset with clean labels. This could be helpful if we'd like to have a dataset view where only clean labels are present, and the rest are filtered out. This dataset can then be passed down to downstream ML frameworks for training.

```python
ds_clean = clean_view(dataset=ds_train, label_issues=label_issues)
```

---

## GSoC Experience
I had an ambiguous high-level problem that I was trying to solve, and I was fortunate that the mentors gave me a lot of freedom to solve this problem. It's not something I can take for granted as I had a high responsibility to my mentors, but it was really rewarding to own the whole technical process from big idea to shipping out the solution. During my GSoC, I found myself drawing on much more than just my experience in software engineering. For example, I utilized my experience in academic research, presentation skills, and writing to execute the project successfully. I learned how to collaborate on a product across other teams, and, perhaps, most importantly, how to take feedback and iterate rapidly. Additionally, I improved my leadership and communication skills by co-leading community efforts. I welcomed new open source contributors to Hub, assigned them tasks, and helped them get started. 

Even though the codebase is immense and unfamiliar to me, I succeeded in this project because I learned to ask the right questions to understand the scope of a problem. Even if I'm unfamiliar with the technology, I can ask strategic questions to get enough understanding to work towards a solution. I also always come back to thinking about how a user might experience a feature or what else they might need or want. This has helped me stay focused on the problem I'm solving, even though a problem was ambiguous. It's also important not to be afraid when facing a new problem. In my day-to-day, I was constantly working on things that were new to me or that I'd never done before. This is just one feature I developed within a large codebase, but it shows how I could start with the high-level goal, carefully consider the technical implications and design a  solution.

---

## Takeaways

If someone who's reading this is interested in participating in GSoC, here are a few advices I can derive from my experience.

- Don't be afraid when facing a new problem. Even if unfamiliar with the technology, learn to ask strategic questions to get enough understanding to work towards a solution. As you work through the problem, clarify the requirements, edge cases and trade-offs. Try to scope out and break down the problem you're working on.

- It helps to develop big-picture skills. Keep high-level use cases and end-users in mind while going layers deeper down to the code to actually implement a feature. It really helps to think about how a user might experience a feature or what else they might need.

- Finally, take ownership of the project you're working on. Be proactive in setting up syncs, come up with discussion points ahead of your mentors, and plan your next steps before the mentors ask you to do so. Communicate your ideas in a clear, concise way so that they could understand where you're at and give you the most help they can. This will help you to take their feedback and iterate rapidly.

---

## Acknowledgements

I want to thank my mentors, the Activeloop community, and Google for giving me an enriching experience this summer. I am incredibly thankful for the opportunity to participate in this program and enhance my programming, communication, and leadership skills while improving the [Hub](https://github.com/activeloopai/Hub) project.

I am immensely grateful to [Davit Buniatyan](https://github.com/davidbuniat)  and  [Fariz Rahman](https://github.com/farizrahman4u)  for their constant guidance, timely and extensive code reviews, and engagement in regular syncs. I would also like to thank  [Mikayel Harutyunyan](https://github.com/mikayelh) for allowing me to take ownership of the community efforts and [Ivo Stranic](https://github.com/istranic) for providing a high-level overview of the product and user needs as I iterated over my solution.

The [Activeloop](https://www.activeloop.ai/) community has always been welcoming and helpful. Their efforts towards improving the developer experience have provided a conducive environment to contribute. I am grateful and thrilled to be a part of it.
