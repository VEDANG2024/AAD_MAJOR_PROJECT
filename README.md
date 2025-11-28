**INSTALL ALL THE DEPENDENCIES FIRST USING PIP
Run in order Phase1-->Training-->Phase3
dataset is GSM8K from huggingface
use TPU environment on kaggle or colab**



**phase 1 output:** 

Loading GSM8K dataset...
Success! <img width="857" height="625" alt="image" src="https://github.com/user-attachments/assets/ee819f4b-c998-4e9d-950d-d7d9d315c2ff" />

**phase2 output:** 

Step    Training Loss
10      2.4512
20      1.8934
30      1.1023
40      0.8541
50      0.6120
Training Complete! Model saved.

 **phase3 console output:** 
 
--- UGAD-Lite Router Demo ---

Query 1: 'What is 2+2?'
  > Uncertainty Score: 0.0012 (Threshold: 1.2)
  > Decision: Route to SLM (Student)
  > Output: "4" (Latency: 45ms)

Query 2: 'Solve this complex logic puzzle...'
  > Uncertainty Score: 1.8450 (Threshold: 1.2)
  > Decision: Route to LLM (Teacher)
  > Output: [Simulated GPT-4 Response] (Latency: 1200ms)
