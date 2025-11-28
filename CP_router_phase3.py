import torch
import numpy as np

# --- The "Secret Sauce": FBE Metric ---
def calculate_fbe_uncertainty(logits, lambda_param=1.0):
    """
    Calculates the Full-Binary Entropy (FBE) score.
    This metric detects if the model is 'confused' between options.
    """
    # Convert raw logits (scores) into probabilities (0.0 to 1.0)
    probs = torch.softmax(logits, dim=-1)
    
    # 1. Full Shannon Entropy (General Confusion)
    # Measures how "flat" the probability distribution is
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    
    # 2. Binary Entropy of the Top Prediction (Specific Confidence)
    # Measures confidence specifically in the #1 choice vs everything else
    top_p, _ = probs.max(dim=-1)
    # Clip values to avoid log(0) errors
    top_p = torch.clamp(top_p, 1e-9, 1.0 - 1e-9)
    binary_entropy = -(top_p * torch.log(top_p) + (1 - top_p) * torch.log(1 - top_p))
    
    # Combined Score: The higher this is, the more uncertain the model is
    return entropy + (lambda_param * binary_entropy)

# --- The Router Class ---
class CPRouter:
    def __init__(self, calibration_threshold=1.5):
        # In a real system, this threshold is calculated from validation data.
        # Here we set it manually for the demo.
        self.threshold = calibration_threshold

    def route(self, logits):
        score = calculate_fbe_uncertainty(logits).item()
        
        print(f"  > Uncertainty Score: {score:.4f} (Threshold: {self.threshold})")
        
        if score > self.threshold:
            return "LLM (Teacher)" # High uncertainty -> Escalate to GPT-4
        else:
            return "SLM (Student)" # Low uncertainty -> Handle locally

# --- DEMO: Simulating the Router in Action ---

# 1. Initialize Router
router = CPRouter(calibration_threshold=1.2)

print("--- UGAD-Lite Router Demo ---\n")

# Simulation 1: An Easy Question (e.g., "2+2=?")
# The model is very confident. One token has huge probability.
print("Query 1: 'What is 2+2?'")
# Logits: [Low, Low, HIGH, Low] -> Confident
easy_logits = torch.tensor([[-2.0, -1.5, 5.0, -2.0]]) 
decision = router.route(easy_logits)
print(f"  > Decision: Route to {decision}\n")

# Simulation 2: A Hard/Ambiguous Question (e.g., Logic Riddle)
# The model is confused. Probabilities are spread out.
print("Query 2: 'Solve this complex logic puzzle...'")
# Logits: [Med, Med, Med, Low] -> Confused/Flat
hard_logits = torch.tensor([[2.0, 2.1, 1.9, 0.5]]) 
decision = router.route(hard_logits)
print(f"  > Decision: Route to {decision}\n")
