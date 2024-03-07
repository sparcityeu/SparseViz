# Tutorial 1: How to Use the Config File

The config file in SparseViz plays a crucial role, allowing the library
to function effectively without the need for writing additional C/C++
code. It facilitates the use of pre-implemented orderings and kernels,
enabling users to visualize matrices and tensors efficiently. This
tutorial aims to guide you through the nuances of effectively utilizing
the config file.

In SparseViz, the config file is seperated into various sections with each of which adding different functionality to the config file.
These sections are prefixed and suffixed with \* to notice the config file reader that they are section headings.
Here are the sections that are currently available for use within SparseViz:

## \*LICENCES\*

This is the section under which every license should be indicated,
like so:

```
RABBIT_LICENCE = rabbit_licence.txt
```

If you lack certain licenses, you should indicate their absence with
"None".

## \*SETTINGS\*

This is perhaps the most important section of the config file. Under
which you should indicate various settings that are going to determine
how your library is going to be initiated. Here are all the settings
that can or should be given under this section, as well as their
explanations as to what they are doing.

### IMPORTANT REMARKS

Settings prefixed with \* indicate that its explicit definition is
mandatory in the config file, otherwise program will prompt a runtime
error.

Settings prefixed with ? indicates that its explicit definition is
mandatory under certain conditions, which will be explained at the end
of this tutorial. The absence of them will prompt a runtime error if
these conditions are met.

### LOG_FILE

```
LOG_FILE = [YOUR CSV PATH]
```

SparseViz provides a performance logger that is accessible upon the
termination of the program to showcase the details of each of the
operations made with sparse data structures. This setting determines
the path of the CSV to which the performance logging is going to be
written.

### TIMING_LOG

```
TIMING_LOG = [YOUR BOOLEAN VALUE] // Default = true
```

In addition to providing a CSV file upon termination, SparseViz is also
able to log the performance values onto the terminal if requested by
this setting. Enabling this is going to make every one of the
performance reports to be logged onto the terminal.

### EXPORT_ORDERED_SPARSE_STRUCTURES

```
EXPORT_ORDERED_SPARSE_STRUCTURES = [YOUR BOOLEAN VALUE] // Default = true
```

As indicated in the introduction, SparseViz can export and save sparse
structures onto files for easy retrieval in further usage. This
setting determines whether sparse structures having gone through some
type of order in the current run should be written into the disk.

### USE_EXISTING_ORDERED_SPARSE_STRUCTURES

```
USE_EXISTING_ORDERED_SPARSE_STRUCTURES = [YOUR BOOLEAN VALUE] // Default = true
```

This setting determines whether ordered sparse structures having been
written into disk previously are allowed to be read in the subsequent
runs.

### EXPORT_ORDERINGS

```
EXPORT_ORDERINGS = [YOUR BOOLEAN VALUE] // Default = true
```

Similar to EXPORT_ORDERED_SPARSE_STRUCTURES, this setting determines whether orderings that have been
made in the current run should be written into the disk.

### USE_EXISTING_ORDERINGS

```
USE_EXISTING_ORDERINGS = [YOUR BOOLEAN VALUE] // Default = true
```

Similar to USE_EXISTING_ORDERED_SPARSE_STRUCTURES, this setting determines whether ordering having written
into disk previously is allowed to be read in the subsequent runs.

### \* PROJECT_DIR

```
PROJECT_DIR = [YOUR PATH TO PROJECT DIRECTORY]
```

This setting should be set to the absolute path of the directory under
which SparseViz is located.

### ? MATRIX_FILES_DIR

```
MATRIX_FILES_DIR = [YOUR PATH TO MATRIX FILES DIRECTORY]
```

This setting is set to the absolute path of the directory under which
your matrix files (ones ending with .mtx or .mtx.bin) are located.

### ? MATRIX_ORDERING_FILES_DIR

```
MATRIX_ORDERING_FILES_DIR = [YOUR PATH TO MATRIX ORDERING FILES DIRECTORY]
```

This setting is set to the absolute path of the directory under which
your matrix ordering files (ones ending with .bin) are located.

### ? MATRIX_VISUALIZATION_FILES_DIR

```
MATRIX_VISUALIZATION_FILES_DIR = [YOUR PATH TO MATRIX VISUALIZATION FILES DIR]
```

This setting is set to the absolute path of the directory into which
matrix visualization files are going to be generated. Upon termination
of the program, all visualization-related outputs are going to be found
there.

### ? TENSOR_FILES_DIR

```
TENSOR_FILES_DIR = [YOUR PATH TO TENSOR FILES DIRECTORY]
```

This setting is set to the absolute path of the directory under which
your tensor files (ones ending with .tns or .tns.bin) are located.

### ? TENSOR_ORDERING_FILES_DIR

```
TENSOR_ORDERING_FILES_DIR = [YOUR PATH TO TENSOR ORDERING FILES DIRECTORY]
```

This setting is set to the absolute path of the directory under which
your tensor ordering files (ones ending with .bin) are located.

### ? TENSOR_VISUALIZATION_FILES_DIR

```
TENSOR_VISUALIZATION_FILES_DIR = [YOUR PATH TO TENSOR VISUALIZATION FILES DIR]
```

This setting is set to the absolute path of the directory into which
tensor visualization files are going to be generated. Upon termination
of the program, all visualization-related outputs are going to be found
there.

### LOGO_PATH

```
LOGO_PATH = [YOUR PATH FOR THE LOGO]
```

This setting is set to the absolute path of the logo that will be used
for the logo of every HTML document generated.

### FAVICON_PATH

```
FAVICON_PATH = [YOUR PATH FOR THE FAVICON]
```

This setting is set to the absolute path of the logo that will be used
for the favicon of every HTML document generated.

### MAX_DIM

```
MAX_DIM = [YOUR INTEGER] // Default = 64
```

This setting is set to an integer representing the maximum dimension
that can be seen in the visualization files.

### \* ZOO_TYPE

```
ZOO_TYPE = [YOUR ZOO TYPE] // Currently availables are: {"MAT", "MATORD", "TENS", "TENSORD", "FULLTENSOR"}
```

This setting is set to one of the 5 available zoo types that SparseViz
currently supports. What each of them does will be explained at the end
of this tutorial.

### \* CHART_TYPE

```
CHART_TYPE = [YOUR CHART TYPE] // Currently availables are: {"NNZ", "ABS"}
```

This setting is set to one of 2 available chart types that SparseViz
currently supports. It determines the chart type that will be used in
the visualization, whether it'll be nonzero or absolute value-based.

## \*MATRICES\*

This section is for providing every one of the .mtx filenames that will
be used throughout the execution of the program. Giving only the
filename with its appropriate extension (probably .mtx) is enough as
their full path can be inferred with the matrix file directory setting
you have indicated in the former section. Every .mtx filename should
allocate a separate line.

## \*MATRIX_ORDERINGS\*

This section provides every matrix ordering that will be used
throughout the execution of the program. The usage should be as
follows:

```
[YOUR MATRIX ORDERING CLASS NAME] [YOUR ORDERING NAME] [YOUR PARAMETERS GIVEN TO THE ORDERING CLASS CONSTRUCTOR SEPARATED WITH '/']
```

Indicating parameters to the ordering class constructor can be skipped
safely, if and only if the ordering constructor does not expect any.
Every ordering definition should allocate a separate line.

## \*TENSORS\*

This section provides every one of the .tns filenames that will be used
throughout the execution of the program. Giving only the filename with
its appropriate extension (probably .tns) is enough as their full path
can be inferred with the tensor file directory setting you have
indicated in the former section. Every .tns filename should allocate a
separate line.

## \*TENSOR_ORDERINGS\*

This section provides every tensor ordering that will be used
throughout the execution of the program. The usage should be as
follows:

```
[YOUR TENSOR ORDERING CLASS NAME] | [YOUR ORDERING NAME] | [YOUR PARAMETERS GIVEN TO THE ORDERING CLASS CONSTRUCTOR SEPARATED WITH '/']
```

Indicating parameters to the ordering class constructor can be skipped
safely, if and only if the ordering constructor does not expect any.
Every ordering definition should allocate a separate line.

## \*MATRIX_KERNELS\*

This section is for providing every CPU matrix kernel that will be
executed on your ordered sparse matrices. The usage should be as
follows:

```
[YOUR KERNEL NAME] | [YOUR THREAD COUNTS] | [YOUR SCHEDULING POLICY] | [YOUR CHUNK SIZE] | [YOUR N_RUN] | [YOUR N_IGNORE]
```

Because we have made matrix kernel parallelization possible with
OpenMP, you are required to provide necessary OMP arguments while
running your kernels. Currently, 3 arguments could be set from the
config file to arrange your parallel environment, which are:

### Setting Thread Counts:

This is used for indicating the number of threads that will be working
during your kernel's execution. If you want to test your kernels
multiple times, with each of which having different thread counts
working on, you can give an array of thread counts directly from
within the config file by separating its elements with a delimiter of
'/', like so: 1/2/4/8/16.

### Setting Scheduling Policy:

OpenMP scheduling policies can be set from the config file as well.
Available scheduling policies to be set are: {"static", "auto",
"dynamic", "guided"}.

### Setting Chunk Size:

One last parameter that can be set from the config file is chunk size.
Its use is simply indicating an integer to determine the chunk size of
the parallel environment.

N_RUN parameter determines the number of times that your kernel is
going to get repeated and N_IGNORE determines the first number of
executions to be ignored. Every matrix kernel definition should
allocate a separate line in the config file.

## \*GPU_MATRIX_KERNELS\*

This section is for providing every GPU matrix kernel that will be
executed on your ordered sparse matrices. The usage should be as
follows:

```
[YOUR KERNEL NAME] | [YOUR GRID SIZE] | [YOUR BLOCK SIZES] | [YOUR N_RUN] | [YOUR N_IGNORE]
```

Because we have made matrix GPU kernel parallelization possible with
Cuda, you are required to provide necessary Cuda arguments while
launching your kernels. Currently, 2 arguments could be set from the
config file to arrange your parallel environment, which are:

### Setting Grid Size:

While launching Cuda kernels, 2 parameters are required to be set. One
of them is the size of the grid which determines the number of blocks
that it is going to include in it. Similar to the CPU Matrix Kernel
thread count argument, the grid size parameter could be given as an
array of grid sizes where each element is separated from one another
with a delimiter of '/'.

### Setting Block Size:

The other argument that Cuda dictates to be set while launching GPU
kernels is block sizes that determine the number of threads that will
be working within each block. Similar to the CPU Matrix Kernel thread
count argument, the block size parameter could be given as an array of
block sizes where each element is separated from one another with a
delimiter of '/'.

One important thing to consider in the above 2 settings is that their
length should be equal to each other -if they are given as an
array-, as they will be zipped while processing.

N_RUN and N_IGNORE are the same as their counterparts in CPU Matrix
Kernels. Every GPU matrix kernel definition should allocate a separate
line in the config file.

## \*TENSOR_KERNELS\*

This section is for providing every CPU tensor kernel that will be
executed on your ordered sparse matrices. The usage should be as
follows:

```
[YOUR KERNEL NAME] | [YOUR THREAD COUNTS] | [YOUR SCHEDULING POLICY] | [YOUR CHUNK SIZE] | [YOUR N_RUN] | [YOUR N_IGNORE]
```

Because we have made tensor kernel parallelization possible with
OpenMP, you are required to provide necessary OMP arguments while
running your kernels. Currently, 3 arguments could be set from the
config file to arrange your parallel environment, which are:

### Setting Thread Counts:

This is used for indicating the number of threads that will be working
during your kernel's execution. If you want to test your kernels
multiple times, with each of which having different thread counts
working on, you can give an array of thread counts directly from
within the config file by separating its elements with a delimiter of
'/', like so: 1/2/4/8/16.

### Setting Scheduling Policy:

OpenMP scheduling policies can be set from the config file as well.
Available scheduling policies to be set are: {"static", "auto",
"dynamic", "guided"}.

### Setting Chunk Size:

One last parameter that can be set from the config file is chunk size.
Its use is simply indicating an integer to determine the chunk size of
the parallel environment.

N_RUN parameter determines the number of times that your kernel is
going to get repeated and N_IGNORE determines the first number of
executions to be ignored. Every tensor kernel definition should
allocate a separate line in the config file.

## \*GPU_TENSOR_KERNELS\*

This section is for providing every GPU tensor kernel that will be
executed on your ordered sparse matrices. The usage should be as
follows:

```
[YOUR KERNEL NAME] | [YOUR GRID SIZE] | [YOUR BLOCK SIZES] | [YOUR N_RUN] | [YOUR N_IGNORE]
```

Because we have made tensor GPU kernel parallelization possible with
Cuda, you are required to provide necessary Cuda arguments while
launching your kernels. Currently, 2 arguments could be set from the
config file to arrange your parallel environment, which are:

### Setting Grid Size:

While launching Cuda kernels, 2 parameters are required to be set. One
of them is the size of the grid which determines the number of blocks
that it is going to include in it. Similar to the CPU Tensor Kernel
thread count argument, the grid size parameter could be given as an
array of grid sizes where each element is separated from one another
with a delimiter of '/'.

### Setting Block Size:

The other argument that Cuda dictates to be set while launching GPU
kernels is block sizes that determine the number of threads that will
be working within each block. Similar to the CPU Tensor Kernel thread
count argument, the block size parameter could be given as an array of
block sizes where each element is separated from one another with a
delimiter of '/'.

One important thing to consider in the above 2 settings is that their
length should be equal to each other -if they are given as an
array-, as they will be zipped while processing.

N_RUN and N_IGNORE are the same as their counterparts in CPU Matrix
Kernels. Every GPU tensor kernel definition should allocate a separate
line in the config file.

## ZOO TYPES

One final remark that is left unmentioned related to the config file is
what each ZOO_TYPE does. ZOO_TYPE determines the library's mode of
running. As said previously, 5 zoo types could be used for now. Each one
of them is explained as follows:

### "MAT"

MAT ZOO_TYPE enables the library to work with matrices. Its use requires
the following config settings to be defined explicitly:

MATRIX_FILES_DIR

MATRIX_ORDERING_FILES_DIR

MATRIX_VISUALIZATION_FILES_DIR

While it would activate the matrix flow of the library, one feature that
is separated from other matrices ZOO_TYPE is in its way of visualizing
matrices.

In the MAT ZOO_TYPE, separate html files are generated for every one of
the matrices whose filename is mentioned under the \*MATRICES\* section.
Orderings implemented on them will be visualized one under the other
within the same file whose name is typically determined by the matrix
name itself.

### "MATORD"

MATORD ZOO_TYPE enables the library to work with matrices as well. Its
use requires the same config settings to be defined explicitly as MAT
ZOO_TYPE:

MATRIX_FILES_DIR

MATRIX_ORDERING_FILES_DIR

MATRIX_VISUALIZATION_FILES_DIR

While it would activate the matrix flow of the library, one feature that
it separated from the other available matrix-based ZOO_TYPE is its
unique way of visualizing matrices.

In the MATORD ZOO_TYPE, separate HTML files are generated for every one
of the orderings whose properties are mentioned under the
\*MATRIX_ORDERINGS\* section. Matrices on which the very same ordering
is implemented are visualized one under the other within the same file
whose name is typically determined by the name of the ordering itself.

### "TENS"

TENS ZOO_TYPE enables the library to work with tensors. Its use requires
the following config settings to be defined explicitly:

TENSOR_FILES_DIR

TENSOR_ORDERING_FILES_DIR

TENSOR_VISUALIZATION_FILES_DIR

While it would activate the tensor flow of the library, one feature that
is separated from other available tensor-based ZOO_TYPEs is in its way
of visualizing tensors.

In the TENS ZOO_TYPE, separate html files are generated for every one of
the tensors whose filename is mentioned under the \*TENSORS\* section.
Orderings implemented on them will be visualized one under the other
within the same file whose name is typically determined by the tensor
name itself.

### "TENSORD"

TENSORD ZOO_TYPE enables the library to work with tensors as well. Its
use requires the same config settings to be defined explicitly as TENS
ZOO_TYPE:

TENSOR_FILES_DIR

TENSOR_ORDERING_FILES_DIR

TENSOR_VISUALIZATION_FILES_DIR

While it would activate the tensor flow of the library, one feature that
it separated from the other available tensor-based ZOO_TYPEs is its
unique way of visualizing matrices.

In the TENSORD ZOO_TYPE, separate HTML files are generated for every one
of the orderings whose properties are mentioned under the
\*TENSOR_ORDERINGS\* section. Tensors on which the very same ordering is
implemented are visualized one under the other within the same file
whose name is typically determined by the name of the ordering itself.

### "FULLTENSOR"

\[TO BE COMPLETED\]

Having set our config file up, we can run our library properly by
executing the executable file of SparseViz with the only argument
pointing to the absolute path of the config file we have built.
