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
- [ ] **Datasets**
    - [X] Add support for the SWDE dataset

### **Models & AI Generation**
- [ ] **Model Integration & Fine-Tuning**
    - [ ] Fine-Tune the generation model
    - [ ] Integrate and test the Gemma 3 270M model
- [ ] **Inference & Optimization**
    - [ ] Add vLLM as an option for AI generation (*Est. 20 mins*)
    - [ ] Implement quantization for models where possible

### **Evaluation**
- [ ] Develop and implement additional evaluation metrics
- [ ] Create a page-level evaluation using MarkupLM on the SWDE dataset
- [ ] Evaluate our custom model/system ("ours")
- [ ] Embedder evaluation

### **System Performance**
- [ ] **(Pending)** Improve overall system speed
- [ ] Logger to log everything, including Speed of each operation, or just use Pheonix

