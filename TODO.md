### **High-Level Goals**
- [ ] Write Paper
- [ ] Run Experiments
- [ ] Schedule Experiments
- [ ] ML Flow or DVC Experiment Tracking

### **Core Development & Cleanup**
- [ ] Iterate through the entire system for a full review
- [ ] Connect the Command-Line Interface (CLI)
- [ ] Connect or delete any loose/unintegrated code
- [ ] Add comprehensive docstrings to all functions and classes
- [ ] Implement a `DEBUG MODE` to profile and identify performance bottlenecks

### **Datasets & Preprocessing**
- [ ] **Postprocessor**
    - [ ] Improve the general pipeline
    - [ ] Implement the specific improvement we previously discussed

### **Models & AI Generation**
- [ ] **Model Integration & Fine-Tuning**
    - [ ] Fine-Tune the generation model
    - [ ] Integrate and test the Gemma 3 270M model
- [ ] Seperate the prompts

### **Evaluation**
- [X] Create a page-level evaluation using MarkupLM on the SWDE dataset
- [ ] Implement ("ours") after you finish getting some results

### **System Performance**
- [ ] **(Pending)** Improve overall system speed
- [ ] Logger to log everything, including Speed of each operation, or just use Pheonix

## **Serious TODO**
- [ ] A way to allow for the pipeline to change (like a config dictionary)
- [ ] Add good schemas for the SWDE dataset
- [ ] way to error analysis 
- [ ] logging / cleaning / refactoring
- [ ] Documentation 