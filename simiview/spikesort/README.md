# SpikeSorter

The spike sorter currently consists of two windows, the main window and the sorting window

# Main Window
The main window is where you may load a file or save in a specific output format, and contains the channel table

## Channel Table

Here you may select a channel to sort and you may also toggle bad channels.  
Bad channels are excluded from the common median computation.  

# Sorting Window

The sorting window is comprised of 5 views:

## The Pointcloud View
The pointcloud view contains a toolbar, with 3 fields which allows you to select the variable plotted along that dimension

The main view contains a 3d pointcloud, here you may use the Lasso Tool to select points to perform certain operations.  The lasso tool is intended to be used in the following manner

1. Select an operation
   1. example: by default, `Control+a`, activates the `add` operation, to add selected points to the active cluster
2. Click down with your mouse to start drawing
3. While holding `Control` to prevent camera interaction, drag with your mouse to draw the lasso polygon
   1. Note the colour of the lasso will reflect the operation you have toggled, while not currently implemented, in the future right clicking in this step will cancel the lasso
4. Release the lasso to close the polygon and perform the relevant operation

In addition to the lasso tool, by holding down `Alt` and dragging your cursor, you may select individual points in the pointcloud. This will highlight the corresponding waveform in the waveform view.

## The Waveform View
This view contains all the waveforms, coloured by their cluster/validity status.

Waveforms will be layered such that the invalid and unsorted waveforms are at the bottom, and waveforms belonging to each cluster are grouped together in increasing order. If a cluster is active, this will be brought to the top.  If a single waveform is active (see holding `Alt` above), it will rendered above all others.

## The Unit View
This view shows mean waveforms for each cluster, and allows some simple operations.  

Activate a cluster by clicking on one of the clusters here.  The active cluster will have a border  
Select multiple clusters for an operation by `Shift` clicking multiple.  Selected clusters will be highlighted.  The last cluster added to the selection will also be active.  Activating a cluster normally deselects all clusters.  
Right click a single or a selection of cluster(s) to open the context menu:  

- Show/Hide: clusters that are hidden will have their transparency set to settings.hiddenTransparency
- Merge: merges selected clusters into the active cluster
- Invalidate: Marks all waveforms in a cluster as invalid
- Delete: Marks all waveforms in a cluster as unsorted

## The CCG view
This view shows the auto/cross correlograms for all pairs of clusters

## The Continuous View
For files where continuous data is available, this is rendered here. 

Controls:
- scrolling navigates through the file
- `Control` + scrolling scales voltage
- `Shift` + scrolling scales time
- clicking sets the threshold
- `f` turns on the frequency filter
- `c` turns on the common median rejection
- `t` detects waveforms with the current parameters and updates the viewer

Note that for viewing and for detection, if the common median rejection is enabled, all other good channels must be loaded.  To do this is quite slow.  To prevent having to do this multiple times, the median is cached until an update is made that changes the channel configuration.  This cache is only done for sections of data that have been viewed or computed on.  As such the first detection or scroll through the data might be slow, but should be fast for all subsequent attempts on all channels.

# Settings

All settings are stored in the settings.json file

# Data model

At the moment, data is stored in a folder beside the original data, with a subfolder for each channel.  Each channel's folder will have npy files for the extracted waveforms, PCA and timestamps when available.  As sorts are performed, a clusters.npy file will be saved storing this information as well.  Values such as peak amplitude, etc., are computed on demand.

# Roadmap
## Features to add 
in no particular order
### general
-  [ ] extract the colours for backgrounds and such into colours.py and use constants
   -  [ ] implement theming in settings.py
-  [ ] Add a reload settings operation which allows you to reload settings on the fly
-  [ ] make file reading more flexible 
-  [ ] allow exporting data in some manner
-  [ ] add transparency values in settings for invalid and hidden waveforms
-  [ ] a filter designer
### pointcloud
- [ ] Allow cancelling of lasso  
-  [ ] implement cluster visibility
   -  [ ] make invisible clusters noninteractive
-  [ ] create operations that will make the camera moved to specified positions like numpad 1 3 7 in blender
-  [ ] restrict the camera view to normalized units and scale data flexibly
-  [ ] recompute PCA based on valid spikes on the fly
### waveform view
-  [ ] implement waveform view interaction
### continuous view
-  [ ] Show detected waveforms, with the right colours on the continuous viewer
### unit view
- [ ] Implement context menu in unit view  
  -  [ ] Add event handlers  
  -  [ ] Fix bug of breaking interaction in pointcloud view
-  [ ] Make sure unit view manager updates clusters that have been emptied
### ccg view
-  [ ] Fix the ccg updating to resize the old widgets
-  [ ] Fix ccg bar plot issue

## Features I may not add
-  [ ] Align waveforms
-  [ ] a command palette?