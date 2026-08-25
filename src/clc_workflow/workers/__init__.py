"""
Scripts that run ON THE COMPUTE NODE, not here.

They are shipped to the cluster by submit.py as dpdispatcher common files and executed
by whatever Python the image provides, which does NOT have clc_workflow installed.  So
they must stay standalone: stdlib plus the scientific stack the image already has, and
no clc_workflow imports.  Anything shared with the local drivers gets duplicated here on
purpose -- that is the cost of the node not having the package.
"""
