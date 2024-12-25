'''
Only want the static variables of this dummy class 
No constructor to prevent instance changes to these variables
'''
class SystemParameters:
    # Default values for static variables (assigned in main)
    sigma = 0
    b = 0
    r = 0
    perturbation = None # Perturbation will be set in main if present
    perturbation_index = 0