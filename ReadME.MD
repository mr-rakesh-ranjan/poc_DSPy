# DSPy - Programming - not prompting


## DSPy is a framework for algorithmically optimizing LM prompts and weights, especially when LMs are used one or more times within a pipeline. 



> To use LMs to build a complex system without DSPy, you generally have to: 
    (1) break the problem down into steps, 
    (2) prompt your LM well until each step works well in isolation, 
    (3) tweak the steps to work well together, 
    (4) generate synthetic examples to tune each step, and 
    (5) use these examples to finetune smaller LMs to cut costs.