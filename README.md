# Transformer Complete Lab (From Scratch)

Academic project for the course Artificial Intelligence Topics
Professor: Dimmy Magalhães
Institution: Faculdade iCEV

## Overview

This laboratory explores the complete architecture of the Transformer model by integrating the components developed in previous labs.

In previous exercises, individual mechanisms such as **Scaled Dot-Product Attention**, **Encoder flow**, and **Decoder mechanisms** were implemented separately.

In this lab, these components are assembled into a full **Encoder–Decoder architecture**, capable of performing a toy sequence generation using simulated tensors.

The objective is to demonstrate how the internal components of a Transformer interact during inference, without training a real model.

---

## Objectives

The main objectives of this lab are:

* Integrate previously implemented attention mechanisms into a complete architecture.
* Implement the **Encoder Block** with Self-Attention, residual connections, and Feed Forward Network.
* Implement the **Decoder Block** with Masked Self-Attention and Cross-Attention.
* Apply **Add & Norm** residual structures.
* Implement an **auto-regressive inference loop** that generates tokens sequentially.

---

## Technologies Used

The project follows the restrictions defined in the laboratory instructions.

* Python 3.x
* NumPy

Deep learning frameworks such as PyTorch, TensorFlow, and Keras were intentionally not used in order to manually implement the core mechanisms of the Transformer architecture.

---

## Project Structure

```
transformer-complete-lab
│
├── attention.py
├── add_norm.py
├── decoder.py
├── encoder.py
├── ffn.py
├── inference.py
├── mask.py
├── transformer.py
└── main.py
```

**Module description:**

* **attention.py** → implementation of Scaled Dot-Product Attention
* **ffn.py** → implementation of the position-wise Feed Forward Network
* **add_norm.py** → residual connection combined with Layer Normalization
* **encoder.py** → Encoder block implementation
* **decoder.py** → Decoder block with Masked Self-Attention and Cross-Attention
* **mask.py** → causal mask used in decoder self-attention
* **transformer.py** → integration of Encoder and Decoder into a complete model
* **inference.py** → auto-regressive generation loop
* **main.py** → script used to run the full architecture

---

## How to Run

1. (Optional) Create and activate a virtual environment.

2. Install the required dependency:

```
pip install numpy
```

3. Run the project:

```
python main.py
```

---

## Example Output

The program generates a toy sequence of tokens using randomly initialized tensors.

Example:

```
TRANSFORMER COMPLETE LAB
Generated tokens: [1, 1287, 842, 2853, 2063, 4358, ...]
```

Because the parameters are randomly initialized, the generated sequence may vary between executions.

---

## Concepts Demonstrated

This project demonstrates several core mechanisms of the Transformer architecture:

* Scaled Dot-Product Attention
* Self-Attention
* Masked Self-Attention
* Cross-Attention
* Residual Connections (Add & Norm)
* Position-wise Feed Forward Network
* Auto-regressive sequence generation

---

## AI-Assisted Complementary Support

During the development of this laboratory, AI-based tools were occasionally used as a complementary resource to clarify specific implementation details and conceptual questions related to the Transformer architecture.

Examples of complementary assistance include:

* Verification of the **Softmax computation across the vocabulary dimension** to ensure numerical stability during token probability generation.
* Minor adjustments in the **causal mask integration within the decoder self-attention step**.
* Small debugging support related to **tensor shape alignment during cross-attention operations**.

All suggestions were **reviewed, validated, and integrated manually** during the implementation process.

Final review and validation of the project were performed by:

**Carlos Murilo Nogueira Portela**

---

## Version

Submission version:

```
v1.0
```

This tag represents the final version submitted for evaluation.
