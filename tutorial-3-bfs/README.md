## Logic
We use a loop to launch kernels at each level of the Graph.

Each kernel launch performs looks for new nodes/vertices (neighbours) at the current level, and updates the distance.

Whenever a new node is discovered, it means we now have another level to explore. We use a flag called 'changed' to track this.

If the flag says changed, it means we need to continue the loop to launch another kernel at the next level. Otherwise flag says no change means that the BFS traversal is complete.