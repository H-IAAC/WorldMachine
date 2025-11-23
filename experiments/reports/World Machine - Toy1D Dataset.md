---
date created: 2025-11-21 17:29:54
date modified: 2025-11-21 17:46:53
---
# World Machine - Toy1D Dataset

_Toy1D_ is a synthetic dataset of one-dimensional time series. The series represents a damped physical system with external influences. We designed it for the study of computational world models, a computational system that makes predictions about the current and future state of a "world" based on the sensory data it receives, but inferring the internal structure of that world. It is designed to be simple, while requiring the ability to infer information about the signal to predict its future. 

World Machine is a research project that investigates the concept and creation of computational world models. These AI systems create internal representations to understand and make predictions about the external world. See the [project page](https://h-iaac.github.io/WorldMachine/) for more information. The project is part of the [H.IAAC](https://hiaac.unicamp.br/en/), the Hub for Artificial Intelligence and Cognitive Architecture, located at the Universidade Estadual de Campinas (UNICAMP), Brazil.

## Artifacts

These are the artifact versions used to generate this dataset, and the dataset itself:

- Code: \[version](release GitHub)+LINK
- Doker Container: BADGE+LINK \[eltoncn/world-machine:ver](link zenodo)
- Dataset: BADGE+LINK

## Dataset Definition

The _Toy1D_ is a synthetic dataset of one-dimensional time series that can be intuitively thought of as an object moving along a line. The series represents a damped physical system, with a state given by:

$$ \vec{x}_{i+1} = clip^{2,3}(F\vec{x}_i+\vec{u}_i)$$

where 
- $\vec{x}_i$ is the system state at a time-step $i$: $\vec{x}_{i} = [p_i, v_i, a_i]$ (position, velocity, acceleration)
- $\vec{u}_i$ is na external influence in the system, described in depth in the following subsection
- $F$ is the system transition matrix, with a time step $\Delta t$ defined as unitary, with a negative entry at row 2, column 1 representing a damping in the system:

$$F =  \begin{bmatrix} 
                1 & \Delta t & \frac{\Delta t^2}{2} \\
                -0.1 \Delta t & 1 & \Delta t \\
                0 & 0 & 1
            \end{bmatrix}$$
- $clip^{2,3}(\circ)$ is a clip operator that operates only in dimensions 1 and 2:

$$clip^{2,3}(\vec{v}) = [v^{(0)}, \min(\max(v^{(1)},1), -1), \min(\max(v^{(2)},1), -1)]$$

We initialize the states $\vec{x}_0$ randomly by a uniform distribution, $\vec{x}_0 \sim Uniform(-1, 1)^{3\times1}$. 

The final dataset only uses the position ($\vec{x}_i^{(0)}$) data in the final sensory channel named _external state_, with size 1. The external state is also referred to as "state decoded" in the dataset code.

We define another sensory channel, measurement_, with size 2, as:

$$\vec{s}_i = \tanh(H \vec{x}_i)$$
$$H \sim Uniform(-1, 1)^{2\times2}$$

We fix $H$ at the start of the dataset generation. Note that, depending on the data scales and the $H$ matrix, the measurement can become very similar to the external state.

Since the dataset is stochastic, we can generate different data by controlling the seed of the random number generator. For each seed, we first generate 10,000 sequences of length 1,000 and then segment them into 40,000 sequences of length 200. Finally, we also scale each sequence to the interval $[-1,1]$. We split the dataset into 60\% for training, 20\% for validation, and 20\% for testing.

We provide 15 dataset variations, generated with different seeds.

A sample of the dataset:

<img src="toy1d_experiment0/toy1d_samples.png" max-height=300 />


### External action u

The external action $\vec{u}_i$ ​is a structured, non-stationary periodic system. We generate it from a combination of randomized square-wave signals with varying periods and phases. One component produces periodic oscillations with slowly varying amplitude, while the other produces sparse pulses aligned with the state magnitude. This creates a structured yet highly diverse input signal that excites the system across a wide range of frequencies and amplitudes.

We construct it using two underlying periodic waveforms: a square wave $v(t)$ and a pulse signal $p(t)$, each generated once at the beginning with randomly chosen periods and phases. The period of each waveform is drawn around a nominal value, with large uniform variability, and the initial phase is also random. As a result, the signals remain periodic but do not repeat in a simple or synchronized manner across dimensions. The square wave $v(t)$ serves as a slowly varying, sign-changing excitation, while the pulse signal $p(t)$, created by taking a discrete difference of a second independent square wave, behaves like a spike train that activates only at switching instants.

At each time step $i$, the control input is formed by sampling the precomputed signals at the corresponding index. The first component of the input is state-dependent and uses the pulse signal to inject short, high-contrast bursts whose amplitude scales with the current state magnitude. Concretely, defining:

$$r_i=\sqrt{\max(x_i)}$$

the first control channel is:

$$u_i^{(1)} = 10\, r_i\, p_i$$

where $p_i$ is the pulse value at the current time index. This makes the impulsive component more energetic when the state is large and quieter when the state is small. The third component combines the smoother periodic signal with a slowly varying random amplitude,

$$u_i^{(3)} = a_i\, v_i, \qquad a_i \sim \mathrm{Uniform}(0.75,1)$$

so that the oscillatory part remains present but varies in strength over time. The middle control channel is unused.

Together, these elements produce an input sequence that is neither purely deterministic nor purely random. It oscillates with drifting amplitude, occasionally delivers sharp impulses, and adjusts part of its strength according to the system’s current state. In effect, $u_i$​ acts as a rich and non-repetitive excitation signal, intended to probe the dynamics across multiple time scales while avoiding simple periodicity.


## How to use the dataset

For using the dataset:
- Download it from the [Zenodo record]()
- Install [_world_machine_ package]() ATUALIZAR LINK
- Use this code to load a dataset variation:

```python
import os
import pickle

from world_machine.data import WorldMachineDataLoader

dataset_path = "toy1d_dataset" #UPDATE TO YOUR PATH
seed = 0 #0-14

path = os.path.join(dataset_path, f"seed_{seed}", "toy1d_datasets.pkl")
with open(path, "rb") as file:
    datasets = pickle.load(file)
    
train_dataset = datasets["train"] #train, val or test
train_dataloader = WorldMachineDataLoader(train_dataset)
```


## Datasheet

We based this datasheet on "Datasheets for Dataset" \[1\]. We tried to answer the questions as thoroughly as possible, but considering limitations due to the synthetic and theoretical nature of our dataset.

### Motivation
_The questions in this section are primarily intended to encourage dataset creators to clearly articulate their reasons for creating the dataset and to promote transparency about funding interests._

For what purpose was the dataset created?**
	We created the Toy1D dataset to evaluate the behaviour and capabilities of a computational world model. We designed the dataset to be simple to use for initial experiments, while requiring the ability to infer information about the signal to predict its future.

 **2. Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?**
	The dataset was created by Elton Cardoso do Nascimento (researcher, MSc student) and Paula Dornhofer Paro Costa (professor, advisor), as part of the World Machine project. Both are members of the Cognitive Architectures research line at [H.IAAC](https://hiaac.unicamp.br/en/), the Hub for Artificial Intelligence and Cognitive Architecture, located at the Universidade Estadual de Campinas (UNICAMP), Brazil.

 **3. Who funded the creation of the dataset?**
	This project was supported by the brazilian Ministry of Science, Technology and Innovations, with resources from Law nº 8,248, of October 23, 1991, within the scope of PPI-SOFTEX, coordinated by Softex and published Arquitetura Cognitiva (Phase 3), DOU 01245.003479/2024 -10

### Composition
_Most of these questions are intended to provide dataset consumers with the information they need to make informed decisions about using the dataset for specific tasks. The answers to some of these questions reveal information about compliance with the EU’s General Data Protection Regulation (GDPR) or comparable regulations in other jurisdictions._

**1.  What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?****
	The dataset consists of time series representing the motion of a theoretical object in one-dimensional space. 

**2.  How many instances are there in total (of each type, if appropriate)?****
	The dataset consists of 10,000 sequences of 1,000 temporal steps.

**3. Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?**
	Since the dataset consists of synthetic data, its potential is theoretically limited only by the numerical representation limit. We provide the dataset along with the generation code, which can theoretically generate all possible samples.

**4. What data does each instance consist of?**
	Each instance consists of a sequence of the "object" position and a measurement obtained through a transformation of the position, velocity, and acceleration vector.

**5. Is there a label or target associated with each instance?**
	The target is the future data at each channel. For post-training evaluation, the target indicated is only the position.

**6. Is any information missing from individual instances?**
	No.

**7. Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?**
	No.

**8. Are there recommended data splits (e.g., training, development/validation, testing)?**
	Yes. The generation code split the dataset into 60\% for training, 20\% for validation, and 20\% for testing. The provided datasets follow this split.

**9. Are there any errors, sources of noise, or redundancies in the dataset?**
	No.

**10. Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?**
	The dataset is self-contained.

**11.  Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)?**
	No

**12. Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?**
	No

### Collection Process
_The answers to questions here may provide information that allow others to reconstruct the dataset without access to it._

**1. How was the data associated with each instance acquired?**
	We synthetized the data using a mathematical model of a theoretical physical system.

**2. What mechanisms or procedures were used to collect the data (e.g., hardware apparatuses or sensors, manual human curation, software programs, software APIs)?**
	A Python script generates the data.

**3. If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?**
	We define a seed for the random number generator and generate the dataset samples.

**4. Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?**
	The two researchers on the project were the only people involved in creating the dataset.

**5. Over what timeframe was the data collected?**
	The program generated all the dataset variations in just a few minutes.

**6. Were any ethical review processes conducted (e.g., by an institutional review board)?**
	No, since the dataset does not involve living beings or data generated by them.

### Preprocessing/cleaning/labeling
_The questions in this section are intended to provide dataset consumers with the information they need to determine whether the “raw” data has been processed in ways that are compatible with their chosen tasks. For example, text that has been converted into a “bag-of-words” is not suitable for tasks involving word order._

**1. Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)**
	No.

**2. Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?**
	Yes, we provide only the "raw" data.

**3. Is the software that was used to preprocess/clean/label the data available?**
	No, because we do not apply pre-processing.

### Uses

**1. Has the dataset been used for any tasks already?**
    Yes, the dataset was used for tasks predicting the next element in the sequence. As part of the World Machine project, we also propose some more specific tasks to be evaluated after model training:
    
- Normal: is a normal autoregressive inference. Note that this differs from the loss calculated during training, as we do not use states that are updated "in parallel" during training; instead, we estimate states sequentially, starting from the null state.
- Use state: inference on previously encoded states, without sensory data. We can calculate this in a single inference step by processing the sequence elements in parallel, since the model already encoded these states. We only evaluate this task at the first 50% of a sequence. If the model were merely taking sensory data and manipulating it to generate output, its performance on this task would be poor.
- Prediction: inference of future states, using several previous encoded states and without sensory data. We use the first 50% of states to evaluate the task in the final 50% of the sequence. 
- Prediction Shallow: inference of future states, using only one previous encoded state and without sensory data. We evaluate this task in the final 50% of the sequence. Prediction Shallow is the most important task, as it directly assesses the model's ability to perform inference with context truncation, which would otherwise incur a quadratic cost in sequence length and is a significant issue with current transformers.
- Prediction Local: inference using local mode, that is, of the next immediate state, using only one previous encoded state, without sensory data.


**2. Is there a repository that links to any or all papers or systems that use the dataset?**
    No. We provide only the link for our [project page](https://h-iaac.github.io/WorldMachine/) that uses the dataset.

**3. What (other) tasks could the dataset be used for?**
    We cannot conceive of any other possible use for the dataset besides the study and evaluation of data forecasting models.

**4. Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?**
    No.

**5. Are there tasks for which the dataset should not be used?**
    No. We consider the dataset harmless and not very useful for other types of projects.

### Distribution

**1. Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?**
    Yes, the dataset is freely available.
    
**2. How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?**
    The dataset is free for download at https://doi.org/10.5281/zenodo.17653221
    
**3. Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?**
    Yes, this work is licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a><img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;">
    
**4. Have any third parties imposed IP-based or other restrictions on the data associated with the instances?**
    No.
    
**5. Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?**
    No.
    
### Maintenance
_These questions are intended to encourage dataset creators to plan for dataset maintenance and communicate this plan with dataset consumers._

**1. Who will be supporting/hosting/maintaining the dataset?**
    The dataset will be hosted by Zenodo.

**2. How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**
    Paula Dornhofer Paro Costa can be contacted by <paulad@unicamp.br>

**3. Is there an erratum?**
    No.

**4. Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?**
    We do not plan any updates.

**5. If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were the individuals in question told that their data would be retained for a fixed period of time and then deleted)?**
	It does not relate to people.

**6. Will older versions of the dataset continue to be supported/hosted/maintained?**
    Yes. 

**7. If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**
    If any updates occur, older versions can be obtained from the same Zenodo record.

## References

\[1\] T. Gebru _et al._, “Datasheets for Datasets,” Dec. 01, 2021, _arXiv_: arXiv:1803.09010. doi: [10.48550/arXiv.1803.09010](https://doi.org/10.48550/arXiv.1803.09010).