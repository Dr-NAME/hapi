import pydoc
from .iso import print_iso, print_iso_id, ISO, ISO_ID

profiles = 'profiles'
def print_profiles():
    print('Profiles available:')
    print('  HT        : PROFILE_HT')
    print('  SDRautian : PROFILE_SDRAUTIAN')
    print('  Rautian   : PROFILE_RAUTIAN')
    print('  SDVoigt   : PROFILE_SDVOIGT')
    print('  Voigt     : PROFILE_VOIGT')
    print('  Lorentz   : PROFILE_LORENTZ')
    print('  Doppler   : PROFILE_DOPPLER')

slit_functions = 'slit_functions'
def print_slit_functions():
    print('  RECTANGULAR : SLIT_RECTANGULAR')
    print('  TRIANGULAR  : SLIT_TRIANGULAR')
    print('  GAUSSIAN    : SLIT_GAUSSIAN')
    print('  DIFFRACTION : SLIT_DIFFRACTION')
    print('  MICHELSON   : SLIT_MICHELSON')
    print('  DISPERSION/LORENTZ : SLIT_DISPERSION')

tutorial='tutorial'
units='units'
index='index'
data='data'
spectra='spectra'
plotting='plotting'
python='python'

python_tutorial_text = \
"""
THIS TUTORIAL IS TAKEN FROM http://www.stavros.io/tutorials/python/
AUTHOR: Stavros Korokithakis


----- LEARN PYTHON IN 10 MINUTES -----


PRELIMINARY STUFF

So, you want to learn the Python programming language but can't find a concise
and yet full-featured tutorial. This tutorial will attempt to teach you Python in 10 minutes.
It's probably not so much a tutorial as it is a cross between a tutorial and a cheatsheet,
so it will just show you some basic concepts to start you off. Obviously, if you want to
really learn a language you need to program in it for a while. I will assume that you are
already familiar with programming and will, therefore, skip most of the non-language-specific stuff.
The important keywords will be highlighted so you can easily spot them. Also, pay attention because,
due to the terseness of this tutorial, some things will be introduced directly in code and only
briefly commented on.


PROPERTIES

Python is strongly typed (i.e. types are enforced), dynamically, implicitly typed (i.e. you don't
have to declare variables), case sensitive (i.e. var and VAR are two different variables) and
object-oriented (i.e. everything is an object).


GETTING HELP

Help in Python is always available right in the interpreter. If you want to know how an object works,
all you have to do is call help(<object>)! Also useful are dir(), which shows you all the object's methods,
and <object>.__doc__, which shows you its documentation string:

>>> help(5)
Help on int object:
(etc etc)

>>> dir(5)
['__abs__', '__add__', ...]

>>> abs.__doc__
'abs(number) -> number

Return the absolute value of the argument.'


SYNTAX

Python has no mandatory statement termination characters and blocks are specified by indentation.
Indent to begin a block, dedent to end one. Statements that expect an indentation level end in a colon (:).
Comments start with the pound (#) sign and are single-line, multi-line strings are used for multi-line comments.
Values are assigned (in fact, objects are bound to names) with the _equals_ sign ("="), and equality testing is
done using two _equals_ signs ("=="). You can increment/decrement values using the += and -= operators respectively
by the right-hand amount. This works on many datatypes, strings included. You can also use multiple variables on one
line. For example:

>>> myvar = 3
>>> myvar += 2
>>> myvar
5

>>> myvar -= 1
>>> myvar
4

\"\"\"This is a multiline comment.
The following lines concatenate the two strings.\"\"\"

>>> mystring = "Hello"
>>> mystring += " world."
>>> print mystring
Hello world.

# This swaps the variables in one line(!).
# It doesn't violate strong typing because values aren't
# actually being assigned, but new objects are bound to
# the old names.
>>> myvar, mystring = mystring, myvar


DATA TYPES

The data structures available in python are lists, tuples and dictionaries.
Sets are available in the sets library (but are built-in in Python 2.5 and later).
Lists are like one-dimensional arrays (but you can also have lists of other lists),
dictionaries are associative arrays (a.k.a. hash tables) and tuples are immutable
one-dimensional arrays (Python "arrays" can be of any type, so you can mix e.g. integers,
strings, etc in lists/dictionaries/tuples). The index of the first item in all array types is 0.
Negative numbers count from the end towards the beginning, -1 is the last item. Variables
can point to functions. The usage is as follows:

>>> sample = [1, ["another", "list"], ("a", "tuple")]
>>> mylist = ["List item 1", 2, 3.14]
>>> mylist[0] = "List item 1 again" # We're changing the item.
>>> mylist[-1] = 3.21 # Here, we refer to the last item.
>>> mydict = {"Key 1": "Value 1", 2: 3, "pi": 3.14}
>>> mydict["pi"] = 3.15 # This is how you change dictionary values.
>>> mytuple = (1, 2, 3)
>>> myfunction = len
>>> print myfunction(mylist)
3


You can access array ranges using a colon (:). Leaving the start index empty assumes the first item,
leaving the end index assumes the last item. Negative indexes count from the last item backwards
(thus -1 is the last item) like so:

>>> mylist = ["List item 1", 2, 3.14]
>>> print mylist[:]
['List item 1', 2, 3.1400000000000001]

>>> print mylist[0:2]
['List item 1', 2]

>>> print mylist[-3:-1]
['List item 1', 2]

>>> print mylist[1:]
[2, 3.14]

# Adding a third parameter, "step" will have Python step in
# N item increments, rather than 1.
# E.g., this will return the first item, then go to the third and
# return that (so, items 0 and 2 in 0-indexing).
>>> print mylist[::2]
['List item 1', 3.14]


STRINGS

Its strings can use either single or double quotation marks, and you can have quotation
marks of one kind inside a string that uses the other kind (i.e. "He said 'hello'." is valid).
Multiline strings are enclosed in _triple double (or single) quotes_ (\"\"\").
Python supports Unicode out of the box, using the syntax u"This is a unicode string".
To fill a string with values, you use the % (modulo) operator and a tuple.
Each %s gets replaced with an item from the tuple, left to right, and you can also use
dictionary substitutions, like so:

>>>print "Name: %s\
Number: %s\
String: %s" % (myclass.name, 3, 3 * "-")

Name: Poromenos
Number: 3
String: ---

strString = \"\"\"This is
a multiline
string.\"\"\"

# WARNING: Watch out for the trailing s in "%(key)s".
>>> print "This %(verb)s a %(noun)s." % {"noun": "test", "verb": "is"}
This is a test.


FLOW CONTROL STATEMENTS

Flow control statements are if, for, and while. There is no select; instead, use if.
Use for to enumerate through members of a list. To obtain a list of numbers,
use range(<number>). These statements' syntax is thus:

rangelist = range(10)
>>> print rangelist
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

>>> for number in rangelist:
        # Check if number is one of
        # the numbers in the tuple.
        if number in (3, 4, 7, 9):
            # "Break" terminates a for without
            # executing the "else" clause.
            break
        else:
            # "Continue" starts the next iteration
            # of the loop. It's rather useless here,
            # as it's the last statement of the loop.
            continue
    else:
        # The "else" clause is optional and is
        # executed only if the loop didn't "break".
        pass # Do nothing

>>> if rangelist[1] == 2:
        print "The second item (lists are 0-based) is 2"
    elif rangelist[1] == 3:
        print "The second item (lists are 0-based) is 3"
    else:
        print "Dunno"

>>> while rangelist[1] == 1:
        pass


FUNCTIONS

Functions are declared with the "def" keyword. Optional arguments are set in
the function declaration after the mandatory arguments by being assigned a default
value. For named arguments, the name of the argument is assigned a value.
Functions can return a tuple (and using tuple unpacking you can effectively return
multiple values). Lambda functions are ad hoc functions that are comprised of
a single statement. Parameters are passed by reference, but immutable types (tuples,
ints, strings, etc) *cannot be changed*. This is because only the memory location of
the item is passed, and binding another object to a variable discards the old one,
so immutable types are replaced. For example:

# Same as def funcvar(x): return x + 1
>>> funcvar = lambda x: x + 1
>>> print funcvar(1)
2

# an_int and a_string are optional, they have default values
# if one is not passed (2 and "A default string", respectively).
>>> def passing_example(a_list, an_int=2, a_string="A default string"):
        a_list.append("A new item")
        an_int = 4
        return a_list, an_int, a_string

>>> my_list = [1, 2, 3]
>>> my_int = 10
>>> print passing_example(my_list, my_int)
([1, 2, 3, 'A new item'], 4, "A default string")

>>> my_list
[1, 2, 3, 'A new item']

>>> my_int
10


CLASSES

Python supports a limited form of multiple inheritance in classes.
Private variables and methods can be declared (by convention, this is not enforced
by the language) by adding at least two leading underscores and at most one trailing
one (e.g. "__spam"). We can also bind arbitrary names to class instances.
An example follows:

>>> class MyClass(object):
        common = 10
        def __init__(self):
            self.myvariable = 3
        def myfunction(self, arg1, arg2):
            return self.myvariable

# This is the class instantiation
>>> classinstance = MyClass()
>>> classinstance.myfunction(1, 2)
3

# This variable is shared by all classes.
>>> classinstance2 = MyClass()
>>> classinstance.common
10

>>> classinstance2.common
10

# Note how we use the class name
# instead of the instance.
>>> MyClass.common = 30
>>> classinstance.common
30

>>> classinstance2.common
30

# This will not update the variable on the class,
# instead it will bind a new object to the old
# variable name.
>>> classinstance.common = 10
>>> classinstance.common
10

>>> classinstance2.common
30

>>> MyClass.common = 50
# This has not changed, because "common" is
# now an instance variable.
>>> classinstance.common
10

>>> classinstance2.common
50

# This class inherits from MyClass. The example
# class above inherits from "object", which makes
# it what's called a "new-style class".
# Multiple inheritance is declared as:
# class OtherClass(MyClass1, MyClass2, MyClassN)
>>> class OtherClass(MyClass):
        # The "self" argument is passed automatically
        # and refers to the class instance, so you can set
        # instance variables as above, but from inside the class.
        def __init__(self, arg1):
            self.myvariable = 3
            print arg1

>>> classinstance = OtherClass("hello")
hello

>>> classinstance.myfunction(1, 2)
3

# This class doesn't have a .test member, but
# we can add one to the instance anyway. Note
# that this will only be a member of classinstance.
>>> classinstance.test = 10
>>> classinstance.test
10


EXCEPTIONS

Exceptions in Python are handled with try-except [exceptionname] blocks:

>>> def some_function():
        try:
            # Division by zero raises an exception
            10 / 0
        except ZeroDivisionError:
            print "Oops, invalid."
        else:
            # Exception didn't occur, we're good.
            pass
        finally:
            # This is executed after the code block is run
            # and all exceptions have been handled, even
            # if a new exception is raised while handling.
            print "We're done with that."

>>> some_function()
Oops, invalid.

We're done with that.


IMPORTING:

External libraries are used with the import [libname] keyword.
You can also use from [libname] import [funcname] for individual functions.
Here is an example:

>>> import random
>>> from time import clock

>>> randomint = random.randint(1, 100)
>>> print randomint
64


FILE I/O

Python has a wide array of libraries built in. As an example, here is how serializing
(converting data structures to strings using the pickle library) with file I/O is used:

>>> import pickle
>>> mylist = ["This", "is", 4, 13327]
# Open the file C:\\binary.dat for writing. The letter r before the
# filename string is used to prevent backslash escaping.
>>> yfile = open(r"C:\\binary.dat", "w")
>>> pickle.dump(mylist, myfile)
>>> myfile.close()

>>> myfile = open(r"C:\\text.txt", "w")
>>> myfile.write("This is a sample string")
>>> myfile.close()

>>> myfile = open(r"C:\\text.txt")
>>> print myfile.read()
'This is a sample string'

>>> myfile.close()

# Open the file for reading.
>>> myfile = open(r"C:\\binary.dat")
>>> loadedlist = pickle.load(myfile)
>>> myfile.close()
>>> print loadedlist
['This', 'is', 4, 13327]


MISCELLANEOUS

    -> Conditions can be chained. 1 < a < 3 checks
       that a is both less than 3 and greater than 1.
    -> You can use del to delete variables or items in arrays.
    -> List comprehensions provide a powerful way to create
       and manipulate lists. They consist of an expression
       followed by a for clause followed by zero or more
       if or for clauses, like so:

>>> lst1 = [1, 2, 3]
>>> lst2 = [3, 4, 5]
>>> print [x * y for x in lst1 for y in lst2]
[3, 4, 5, 6, 8, 10, 9, 12, 15]

>>> print [x for x in lst1 if 4 > x > 1]
[2, 3]

# Check if a condition is true for any items.
# "any" returns true if any item in the list is true.
>>> any([i % 3 for i in [3, 3, 4, 4, 3]])
True

# This is because 4 % 3 = 1, and 1 is true, so any()
# returns True.

# Check for how many items a condition is true.
>>> sum(1 for i in [3, 3, 4, 4, 3] if i == 4)
2

>>> del lst1[0]
>>> print lst1
[2, 3]

>>> del lst1



    -> Global variables are declared outside of functions
       and can be read without any special declarations,
       but if you want to write to them you must declare them
       at the beginning of the function with the "global" keyword,
       otherwise Python will bind that object to a new local
       variable (be careful of that, it's a small catch that can
       get you if you don't know it). For example:

>>> number = 5

>>> def myfunc():
        # This will print 5.
        print number

>>> def anotherfunc():
        # This raises an exception because the variable has not
        # been bound before printing. Python knows that it an
        # object will be bound to it later and creates a new, local
        # object instead of accessing the global one.
        print number
        number = 3

>>> def yetanotherfunc():
        global number
        # This will correctly change the global.
        number = 3


EPILOGUE

This tutorial is not meant to be an exhaustive list of all (or even a subset) of Python.
Python has a vast array of libraries and much much more functionality which you will
have to discover through other means, such as the excellent book Dive into Python.
I hope I have made your transition in Python easier. Please leave comments if you believe
there is something that could be improved or added or if there is anything else
you would like to see (classes, error handling, anything).

"""
def print_python_tutorial():
    pydoc.pager(python_tutorial_text)

data_tutorial_text = \
"""

ACCESS YOUR DATA!

Welcome to tutorial on retrieving and processing the data from HITRANonline.


  ///////////////
 /// PREFACE ///
///////////////

HITRANonline API is a set of routines in Python which is aimed to
provide a remote access to functionality and data given by a new project
HITRANonline (http://hitranazure.cloudapp.net).

At the present moment the API can download, filter and process data on
molecular and atomic line-by-line spectra which is provided by HITRANonline portal.

One of the major purposes of introducing API is extending a functionality
of the main site, particularly providing a possibility to calculate several
types of high- and low-resolution spectra based on a flexible HT lineshape.

Each feature of API is represented by a Python function with a set of parameters
providing a flexible approach to the task.


  ///////////////////////
 /// FEATURE SUMMARY ///
///////////////////////

1) Downloading line-by-line data from the HITRANonline site to local database.
2) Filtering and processing the data in SQL-like fashion.
3) Conventional Python structures (lists, tuples, dictionaries) for representing
   a spectroscopic data.
4) Possibility to use a large set of third-party Python libraries to work with a data
5) Python implementation of an HT (Hartmann-Tran [1]) lineshape which is used in spectra.
   simulations. This lineshape can also be reduced to a number of conventional
   line profiles such as Gaussian (Doppler), Lorentzian, Voigt, Rautian,
   Speed-dependent Voigt and Rautian.
6) Python implementation of total internal partition sums (TIPS-2011 [2])
   which is used in spectra simulations.
7) High-resolution spectra simulation accounting pressure,
   temperature and optical path length. The following spectral functions
   can be calculated:
      a) absorption coefficient
      b) absorption spectrum
      c) transmittance spectrum
      d) radiance spectrum
8) Low-resolution spectra simulation using a number of apparatus functions.
9) Possibility to extend with the user's functionality by adding custom lineshapes,
   partitions sums and apparatus functions.

References:

[1] N.H. Ngo, D. Lisak, H. Tran, J.-M. Hartmann.
    An isolated line-shape model to go beyond the Voigt profile in
    spectroscopic databases and radiative transfer codes.
    JQSRT, Volume 129, November 2013, Pages 89–100
    http://dx.doi.org/10.1016/j.jqsrt.2013.05.034

[2] A. L. Laraia, R. R. Gamache, J. Lamouroux, I. E. Gordon, L. S. Rothman.
    Total internal partition sums to support planetary remote sensing.
    Icarus, Volume 215, Issue 1, September 2011, Pages 391–400
    http://dx.doi.org/10.1016/j.icarus.2011.06.004

_______________________________________________________________________


This tutorial will give you an insight of how to use HAPI for Python.

First, let's choose a folder for our local database. Every time you start
your Python project, you have to specify explicitly the name of the
database folder.

>>> db_begin('data')

So, let's download some data from the server and do some processing on it.
Suppose that we want to get line by line data on the main isotopologue of H2O.

For retrieving the data to the local database, user have to specify the following parameters:
1) Name of the local table which will store the downloaded data.
2) Either a pair of molecule and isotopologue HITRAN numbers (M and I),
   or a "global" isotopologue ID (iso_id).
3) Wavenumber range (nu_min and nu_max)

N.B. If you specify the name which already exists in the database,
the existing table with that name will be overrided.

To get additional information on function fetch,
call getHelp:

>>> getHelp(fetch)
...

To download the data, simply call the function "fetch".
This will establish a connection with the main server and get the data using
the parameters listed above:

>>> fetch('H2O', 1, 1, 3400, 4100)
BEGIN DOWNLOAD: H2O
  65536 bytes written to data/H2O.data
  65536 bytes written to data/H2O.data
  65536 bytes written to data/H2O.data
...
  65536 bytes written to data/H2O.data
  65536 bytes written to data/H2O.data
  65536 bytes written to data/H2O.data
Header written to data/H2O.header
END DOWNLOAD
                     Lines parsed: 7524
PROCESSED

The output is shown right after the console line ">>>".
To check the file that you've just downloaded you can open the database
folder. The new plain text file should have a name "H2O.data" and
it should contain line-by-line data in HITRAN format.

N.B. If we want several isotopologues in one table, we should
use fetch_by_ids instead of just fetch. Fetch_by_ids takes a "global"
isotopologue ID numbers as an input instead of HITRAN's "local" identification.
See getHelp(fetch_by_ids) to get more information on this.

To get a list of tables which are already in the database,
use tableList() function (it takes no arguments):
>>> tableList()

To learn about the table we just downloaded, let's use a function "describeTable".

>>> describeTable('H2O')
-----------------------------------------
H2O summary:
-----------------------------------------
Comment:
Contains lines for H2(16O)
 in 3400.000-4100.000 wavenumber range
Number of rows: 7524
Table type: column-fixed
-----------------------------------------
            PAR_NAME           PAR_FORMAT

            molec_id                  %2d
        local_iso_id                  %1d
                  nu               %12.6f
                  sw               %10.3E
                   a               %10.3E
           gamma_air                %5.4f
          gamma_self                %5.3f
              elower               %10.4f
               n_air                %4.2f
           delta_air                %8.6f
 global_upper_quanta                 %15s
 global_lower_quanta                 %15s
  local_upper_quanta                 %15s
  local_lower_quanta                 %15s
                ierr                  %6s
                iref                 %12s
    line_mixing_flag                  %1s
                  gp                %7.1f
                 gpp                %7.1f
-----------------------------------------

This output tells how many rows are currenty in the table H2O, which
wavenumber range was used by fetch(). Also this gives a basic information
about parameters stored in the table.

So, having the table downloaded, one can perform different operations on it
using API.

Here is a list of operations currently available with API:
1) FILTERING
2) OUTPUTTING
3) SORTING
4) GROUPING


  ////////////////////////////////
 /// FILTERING AND OUTPUTTING ///
////////////////////////////////

The table data can be filtered with the help of select() function.

Use simple select() call to output the table content:

>>> select('H2O')
MI          nu         S         A gair gsel        E_nair    dair  ...
11 1000.288940 1.957E-24 2.335E-02.07100.350 1813.22270.680.008260  ...
11 1000.532321 2.190E-28 1.305E-05.04630.281 2144.04590.39-.011030  ...
...

This will display the list of line parameters containing in the table "H2O".

That's the simplest way of using the function select(). Full information
on control parameters can be obtained via getHelp(select) statement.

Suppose that we need a lines from a table within some wavenumber range.
That's what filtering is for. Let's apply a simple range filter on a table.

>>> select('H2O', Conditions=('between', 'nu', 4000, 4100))
MI          nu         S         A gair gsel        E_nair    dair
 11 4000.188800 1.513E-25 1.105E-02.03340.298 1581.33570.51-.013910 ...
 11 4000.204070 3.482E-24 8.479E-03.08600.454  586.47920.61-.007000 ...
 11 4000.469910 3.268E-23 1.627E+00.05410.375 1255.91150.56-.013050 ...
......

As a result of this operation, we see a list of lines of H2O table,
whose wavenumbers lie between 4000 cm-1 and 4100 cm-1.
The condition is taken as an input parameter to API function "select".

To specify a subset of columns to display, use another control parameter -
ParameterNames:

>>> select('H2O', ParameterNames=('nu', 'sw'), Conditions=('between', 'nu', 4000, 4100))

The usage of ParameterNames is outlined below in the section "Specifying a list
of parameters". So far it worth mentioning that this parameter is a part
of a powerful tool for displaying and processing tables from database.

In the next section we will show how to create quieries
with more complex conditions.


  ////////////////////////////
 /// FILTERING CONDITIONS ///
////////////////////////////

Let's analyze the last example of filtering. Condition input variable is
as follows:

                    ('between', 'nu', 4000, 4100)

Thus, this is a python list (or tuple), containing logical expressions
defined under column names of the table. For example, 'nu' is a name of
the column in 'H2O' table, and this column contains a transition wavenumber.
The structure of a simple condition is as follows:

                    (OPERATION, ARG1, ARG2, ...)

Where OPERATION must be in a set of predefined operations (see below),
and ARG1, ARG2 etc. are the arguments for this operation.
Conditions can be nested, i.e. ARG can itself be a condition (see examples).
The following operations are available in select (case insensitive):


DESCRIPTION                   LITERAL                     EXAMPLE
---------------------------------------------------------------------------------
Range:               'RANGE', 'BETWEEN':         ('BETWEEN', 'nu', 0, 1000)
Subset:              'IN', 'SUBSET':             ('IN', 'local_iso_id', [1, 2, 3, 4])
And:                 '&', '&&', 'AND':            ('AND', ('<', 'nu', 1000), ('>', 'nu', 10))
Or:                  '|', '||', 'OR':             ('OR', ('>', 'nu', 1000), ('<', 'nu', 10))
Not:                 '!', 'NOT':                 ('NOT', ('IN', 'local_iso_id', [1, 2, 3]))
Less than:           '<', 'LESS', 'LT':                 ('<', 'nu', 1000)
More than:           '>', 'MORE', 'MT':                 ('>', 'sw', 1.0e-20)
Less or equal than:  '<=', 'LESSOREQUAL', 'LTE':        ('<=', 'local_iso_id', 10)
More or equal than   '>=', 'MOREOREQUAL', 'MTE':        ('>=', 'sw', 1e-20)
Equal:               '=', '==', 'EQ', 'EQUAL', 'EQUALS':  ('<=', 'local_iso_id', 10)
Not equal:           '!=', '<>', '~=', 'NE', 'NOTEQUAL':  ('!=', 'local_iso_id', 1)
Summation:           '+', 'SUM':                 ('+', 'v1', 'v2', 'v3')
Difference:          '-', 'DIFF':                ('-', 'nu', 'elow')
Multiplication:      '*', 'MUL':                 ('*', 'sw', 0.98)
Division:            '/', 'DIV':                 ('/', 'A', 2)
Cast to string:      'STR', 'STRING':            ('STR', 'some_string')
Cast to Python list  'LIST':                    ('LIST', [1, 2, 3, 4, 5])
Match regexp         'MATCH', 'LIKE':            ('MATCH', '\w+', 'some string')
Search single match: 'SEARCH':                  ('SEARCH', '\d \d \d', '1 2 3 4')
Search all matches:  'FINDALL':                 ('FINDALL', '\d', '1 2 3 4 5')
Count within group:  'COUNT' :                  ('COUNT', 'local_iso_id')
---------------------------------------------------------------------------------

Let's create a query with more complex condition. Suppese that we are
interested in all lines between 3500 and 4000 with 1e-19 intensity cutoff.
The query will look like this:

>>> Cond = ('AND', ('BETWEEN', 'nu', 3500, 4000), ('>=', 'Sw', 1e-19))
>>> select('H2O', Conditions=Cond, DestinationTableName='tmp')

Here, apart from other parameters, we have used a new parameter
DestinationTableName. This parameter contains a name of the table
where we want to put a result of the query. Thus we have chosen
a name 'tmp' for a new table.


  ////////////////////////////////////
 /// ACCESSING COLUMNS IN A TABLE ///
////////////////////////////////////

To get an access to particular table column (or columns) all we need
is to get a column from a table and put it to Python variable.

For this purpose, there exist two functions:

  getColumn(...)
  getColumns(...)

The first one returns just one column at a time. The second one returns
a list of solumns.

So, here are some examples of how to use both:

>>> nu1 = getColumn('H2O', 'nu')
>>> nu2, sw2 = getColumns('H2O', ['nu', 'sw'])

N.B. If you don't remember exact names of columns in a particular table,
use describeTable to get an info on it's structure!


  ///////////////////////////////////////
 /// SPECIFYING A LIST OF PARAMETERS ///
///////////////////////////////////////

Suppose that we want not only select a set of parameters/columns
from a table, but do a certain transformations with them (for example,
multiply column on a coefficient, or add one column to another etc...).
We can make it in two ways. First, we can extract a column from table
using one of the functions (getColumn or getColumns) and do the rest
in Python. The second way is to do it on the level of select.
The select function has a control parameter "ParameterNames", which
makes it possible to specify parameters we want to be selected,
and evaluate some simple arithmetic expressions with them.

Assume that we need only wavenumber and intensity from H2O table.
Also we need to scale an intensity to the unitary abundance. To do so,
we must divide an 'sw' parameter by it's natural abundance (0.99731) for
principal isotopologue of water).

Thus, we have to select two columns:
wavenumber (nu) and scaled intensity (sw/0.99731)
>>> select('H2O', )


  ////////////////////////////
 /// SAVING QUERY TO DISK ///
////////////////////////////

To quickly save a result of a query to disk, the user can take an
advantage of an additional parameter "File".
If this parameter is presented in function call, then the query is
saved to file with the name which was specified in "File".

For example, select all lines from H2O and save the result in file 'H2O.txt':
>>> select('H2O', File='H2O.txt')


  ////////////////////////////////////////////
 /// GETTING INFORMATION ON ISOTOPOLOGUES ///
////////////////////////////////////////////

API provides the following auxillary information about isotopologues
present in HITRAN. Corresponding functions use the standard HITRAN
molecule-isotopologue notation:

1) Natural abundances
>>> abundance(mol_id, iso_id)

2) Molecular masses
>>> molecularMass(mol_id, iso_id)

3) Molecule names
>>> moleculeName(mol_id, iso_id)

4) Isotopologue names
>>> isotopologueName(mol_id, iso_id)

5) ISO_ID
>>> getHelp(ISO_ID)

The latter is a dictionary, which contain all information about
isotopologues concentrated in one place.

"""
def print_data_tutorial():
    pydoc.pager(data_tutorial_text)

spectra_tutorial_text = \
"""

CALCULATE YOUR SPECTRA!

Welcome to tutorial on calculating a spectra from line-by-line data.


  ///////////////
 /// PREFACE ///
///////////////

This tutorial will demonstrate how to use different lineshapes and partition
functions, and how to calculate synthetic spectra with respect to different
instruments. It will be shown how to combine different parameters of spectral
calculation to achieve better precision and performance for cross sections.

API provides a powerful tool to calculate cross-sections based on line-by-line
data containing in HITRAN. This features:

*) Python implementation of an HT (Hartmann-Tran [1]) lineshape which is used in
   spectra simulations. This lineshape can also be reduced to a number of
   conventional    line profiles such as Gaussian (Doppler), Lorentzian, Voigt,
   Rautian, Speed-dependent Voigt and Rautian.
*) Python implementation of total internal partition sums (TIPS-2011 [2])
   which is used in spectra simulations.
*) High-resolution spectra simulation accounting pressure,
   temperature and optical path length. The following spectral functions
   can be calculated:
      a) absorption coefficient
      b) absorption spectrum
      c) transmittance spectrum
      d) radiance spectrum
*) Low-resolution spectra simulation using a number of apparatus functions.
*) Possibility to extend with the user's functionality by adding custom lineshapes,
   partitions sums and apparatus functions.
*) An approach to function code is aimed to be flexible enough yet hopefully
   intuitive.

References:

[1] N.H. Ngo, D. Lisak, H. Tran, J.-M. Hartmann.
    An isolated line-shape model to go beyond the Voigt profile in
    spectroscopic databases and radiative transfer codes.
    JQSRT, Volume 129, November 2013, Pages 89–100
    http://dx.doi.org/10.1016/j.jqsrt.2013.05.034

[2] A. L. Laraia, R. R. Gamache, J. Lamouroux, I. E. Gordon, L. S. Rothman.
    Total internal partition sums to support planetary remote sensing.
    Icarus, Volume 215, Issue 1, September 2011, Pages 391–400
    http://dx.doi.org/10.1016/j.icarus.2011.06.004


  ///////////////////////////
 /// USING LINE PROFILES ///
///////////////////////////

Several lineshape (line profile) families are currently available:
1) Gaussian (Doppler) profile
2) Lorentzian profile
3) Voigt profile
4) Speed-dependent Voigt profile
5) Rautian profile
6) Speed-dependent Rautian profile
7) HT profile (Hartmann-Tran)

Each profile has it's own uniwue set of parameters. Normally one should
use profile parameters only in conjunction with their "native" profiles.

So, let's start exploring the available profiles using getHelp:
>>> getHelp(profiles)
Profiles available:
  HTP       : PROFILE_HT
  SDRautian : PROFILE_SDRAUTIAN
  Rautian   : PROFILE_RAUTIAN
  SDVoigt   : PROFILE_SDVOIGT
  Voigt     : PROFILE_VOIGT
  Lorentz   : PROFILE_LORENTZ
  Doppler   : PROFILE_DOPPLER

Output gives all available profiles. We can get additional info on each
of them just by calling getHelp(ProfileName):
>>> getHelp(PROFILE_HT)

Line profiles, adapted for using with HAPI, are written in Python and
heavily using the numerical library "Numpy". This means that the user
can calculate multiple values of particular profile at once having just
pasted a numpy array as a wavenumber grid (array). Let's give a short
example of how to calculate HT profile on a numpy array.

>>> from numpy import arange
    w0 = 1000.
    GammaD = 0.005
    Gamma0 = 0.2
    Gamma2 = 0.01 * Gamma0
    Delta0 = 0.002
    Delta2 = 0.001 * Delta0
    nuVC = 0.2
    eta = 0.5
    Dw = 1.
    ww = arange(w0-Dw, w0+Dw, 0.01)  # GRID WITH THE STEP 0.01
    l1 = PROFILE_HT(w0, GammaD, Gamma0, Gamma2, Delta0, Delta2, nuVC, eta, ww)[0]
    # now l1 contains values of HT profile calculates on the grid ww

On additional information about parameters see getHelp(PROFILE_HT).

It worth noting that PROFILE_HT returns 2 entities: real and imaginary part
of lineshape (as it described in the article given in preface). Apart from
HT, all other profiles return just one entity (the real part).


  ////////////////////////////
 /// USING PARTITION SUMS ///
////////////////////////////

As it was mentioned in the preface to this tutorial, the partition sums
are taken from the TIPS-2011 (the link is given above). Partition sums
are taken for those isotopologues, which are present in HITRAN and in
TIPS-2011 simultaneousely.

N.B. Partition sums are omitted for the following isotopologues which
are in HITRAN at the moment:

ID       M     I         ISO                MOL
--------------------------------------------------
117      12    2     H(15N)(16O)3           HNO3
110      14    2     D(19F)                 HF
107      15    3     D(35Cl)                HCl
108      15    4     D(37Cl)                HCl
111      16    3     D(79Br)                HBr
112      16    4     D(81Br)                HBr
113      17    2     D(127I)                HI
118      22    2     (14N)(15N)             N2
119      29    2     (13C)(16O)(19F)2       COF2
 86      34    1     (16O)                  O
 92      39    1     (12C)H3(16O)H          CH3OH
114      47    1     (32S)(16O)3            SO3
--------------------------------------------------

The data on these isotopologues is not present in TIPS-2011 but is
present in HITRAN. We're planning to add these molecules after TIPS-2013
is released.

To calculate a partition sum for most of the isotopologues in HITRAN,
we will use a function partitionSum (use getHelp for detailed info).
Let's just mention that
The syntax is as follows: partitionSum(M, I, T), where M, I - standard
HITRAN molecule-isotopologue notation, T - definition of temperature
range.

Usecase 1: temperatuer is defined by a list:
>>> Q = partitionSum(1, 1, [70, 80, 90])

Usecase 2: temperature is defined by bounds and the step:
>>> T, Q = partiionSum(1, 1, [70, 3000], step=1.0)

In the latter example we calculate a partition sum on a range of
temperatures from 70K to 3000K using a step 1.0 K, and having arrays
of temperature (T) and partition sum (Q) at the output.


  ///////////////////////////////////////////
 /// CALCULATING ABSORPTION COEFFICIENTS ///
///////////////////////////////////////////

Currently API can calculate the following spectral function at arbitrary
thermodynamic parameters:

1) Absorption coefficient
2) Absorption spectrum
3) Transmittance spectrum
4) Radiance spectrum

All these functions can be calculated with or without accounting of
an instrument properties (apparatus function, resolution, path length etc...)

As it well known, the spectral functions such as absorption,
transmittance, and radiance spectra, are calculated on the basis
of the absorption coefficient. By that resaon, absorption coefficient
is the most important part of simulating a cross section. This part of
tutorial is devoted to demonstration how to calculate absorption
coefficient from the HITRAN line-by-line data. Here we give a brief
insight on basic parameters of calculation procedure, talk about some
useful practices and precautions.

To calculate an absorption coefficient, we can use one of the following
functions:

-> absorptionCoefficient_HT
-> absorptionCoefficient_Voigt
-> absorptionCoefficient_Lorentz
-> absorptionCoefficient_Doppler

Each of these function calculates cross sections using different
lineshapes (the names a quite self-explanatory).
You can get detailed information on using each of these functions
by calling getHelp(function_name).

Let's look more closely to the cross sections based on the Lorentz profile.
For doing that, let's have a table downloaded from HITRANonline.

# get data on CO2 main isotopologue in the range 2000-2100 cm-1
>>> fetch('CO2', 2, 1, 2000, 2100)

OK, now we're ready to run a fast example of how to calculate an
absorption coefficient cross section:

>>> nu, coef = absorptionCoefficient_Lorentz(SourceTables='CO2')

This example calculates a Lorentz cross section using the whole set of
lines in the "co2" table. This is the simplest possible way to use these
functions, because major part of parameters bound to their default values.

If we have matplotlib installed, then we can visualize it using a plotter:
>>> from pylab import plot
>>> plot(nu, coef)

API provides a flexible control over a calculation procedure. This control
can be achieved by using a number of input parameters. So, let's dig
into the depth of the settings.

The input parameters of absorptionCoefficient_Lorentz are as follows:

Name                          Default value
-------------------------------------------------------------------
SourceTables                  '__BUFFER__'
Components                    All isotopologues in SourceTables
partitionFunction             PYTIPS
Environment                   {'T':296., 'p':1.}
WavenumberRange               depends on Components
WavenumberStep                0.01 cm-1
WavenumberWing                10 cm-1
WavenumberWingHW              50 HWHMs
IntensityThreshold            0 cm/molec
GammaL                        'gamma_air'
HITRAN_units                  True
File                          None
Format                        '%e %e'
-------------------------------------------------------------------

Newt we'll give a brief explanation for each parameter. After each description
we'll make some notes about the usage of the correspondent parameter.


SourceTables:     (required parameter)

  List of source tables to take line-by-line data from.
  NOTE: User must provide at least one table in the list.

Components:    (optional parameter)

  List of tuples (M, I, D) to consider in cross section calculation.
  M here is a molecule number, I is an isotopologue number,
  D is an abundance of the component.
  NOTE: If this input contains more than one tuple, then the output
        is an absorption coefficient for mixture of corresponding gases.
  NOTE2: If omitted, then all data from the source tables is involved.

partitionFunction:    (optional parameter)

  Instance of partition function of the following format:
  Func(M, I, T), where Func - numae of function, (M, I) - HITRAN numbers
  for molecule and isotopologue, T - temperature.
  Function must return only one output - value of partition sum.
  NOTE: Deafult value is PYTIPS - python version of TIPS-2011

Environment:    (optional parameter)

  Python dictionary containing value of pressure and temperature.
  The format is as follows: Environment = {'p':pval, 'T':tval},
  where "pval" and "tval" are corresponding values in atm and K
  respectively.
  NOTE: Default value is {'p':1.0, 'T':296.0}

WavenumberRange:    (optional parameter)

  List containing minimum and maximum value of wavenumber to consider
  in cross-section calculation. All lines that are out of htese bounds
  will be skipped. The firmat is as follows: WavenumberRange=[wn_low, wn_high]
  NOTE: If this parameter os skipped, then min and max are taken
  from the data from SourceTables. Deprecated name is OmegaRange.

WavenumberStep:    (optional parameter)

  Value for the wavenumber step.
  NOTE: Default value is 0.01 cm-1.
  NOTE2: Normally user would want to take the step under 0.001 when
         calculating absorption coefficient with Doppler profile
         because of very narrow spectral lines. Deprecated name is OmegaStep.

WavenumberWing:    (optional parameter)

  Absolute value of the line wing in cm-1, i.e. distance from the center
  of each line to the most far point where the profile is considered
  to be non zero. Deprecated name is OmegaStep.
  NOTE: if omitted, then only OmegaWingHW is taken into account.

WavenumberWingHW:    (optional parameter)

  Relative value of the line wing in halfwidths. Deprecated name is OmegaWingHW.
  NOTE: The resulting wing is a maximum value from both OmegaWing and
  OmegaWingHW.

IntensityThreshold:    (optional parameter)

  Absolute value of minimum intensity in cm/molec to consider.
  NOTE: default value is 0.

GammaL:    (optional parameter)

  This is the name of broadening parameter to consider a "Lorentzian"
  part in the Voigt profile. In the current 160-char format there is
  a choise between "gamma_air" and "gamma_self".
  NOTE: If the table has custom columns with a broadening coefficients,
        the user can specify the name of this column in GammaL. This
        would let the function calculate an absorption with custom
        broadening parameter.

HITRAN_units:    (optional parameter)

  Logical flag for units, in which the absorption coefficient shoould be
  calculated. Currently, the choises are: cm^2/molec (if True) and
  cm-1 (if False).
  NOTE: to calculate other spectral functions like transmitance,
  radiance and absorption spectra, user should set HITRAN_units to False.

File:    (optional parameter)

  The name of the file to save the calculated absorption coefficient.
  The file is saved only if this parameter is specified.

Format:    (optional parameter)

  C-style format for the text data to be saved. Default value is "%e %e".
  NOTE: C-style output format specification (which are mostly valid for Python)
        can be found, for instance, by the link:
  http://www.gnu.org/software/libc/manual/html_node/Formatted-Output.html


N.B. Other functions such as absorptionCoefficient_Voigt(_HT, _Doppler) have
identical parameter sets so the description is the same for each function.


  ///////////////////////////////////////////////////////////////////
 /// CALCULATING ABSORPTION, TRANSMITTANCE, AND RADIANCE SPECTRA ///
///////////////////////////////////////////////////////////////////

Let's calculate an absorption, transmittance, and radiance
spectra on the basis of apsorption coefficient. In order to be consistent
with internal API's units, we need to have an absorption coefficient cm-1:

>>> nu, coef = absorptionCoefficient_Lorentz(SourceTables='CO2', HITRAN_units=False)

To calculate absorption spectrum, use the function absorptionSpectrum():
>>> nu, absorp = absorptionSpectrum(nu, coef)

To calculate transmittance spectrum, use function transmittanceSpectrum():
>>> nu, trans = transmittanceSpectrum(nu, coef)

To calculate radiance spectrum, use function radianceSpectrum():
>>> nu, radi = radianceSpectrum(nu, coef)


The last three commands used a default path length (1 m).
To see complete info on all three functions, look for section
"calculating spectra" in getHelp()

Generally, all these three functions use similar set of parameters:

Wavenumber:       (required parameter)

  Wavenumber grid to for spectrum. Deprecated name is Omegas.

AbsorptionCoefficient        (optional parameter)

  Absorption coefficient as input.

Environment={'T': 296.0, 'l': 100.0}       (optional parameter)

  Environmental parameters for calculating  spectrum.
  This parameter is a bit specific for each of functions:
  For absorptionSpectrum() and transmittanceSpectrum() the default
  value is as follows: Environment={'l': 100.0}
  For transmittanceSpectrum() the default value, besides path length,
  contains a temperature: Environment={'T': 296.0, 'l': 100.0}
  NOTE: temperature must be equal to that which was used in
  absorptionCoefficient_ routine!

File         (optional parameter)

  Filename of output file for calculated spectrum.
  If omitted, then the file is not created.

Format        (optional parameter)

  C-style format for spectra output file.
  NOTE: Default value is as follows: Format='%e %e'


  ///////////////////////////////////////
 /// APPLYING INSTRUMENTAL FUNCTIONS ///
///////////////////////////////////////

For comparison of the theoretical spectra with the real-world
instruments output it's necessary to take into account instrumental resolution.
For this purpose HAPI has a function convolveSpectrum() which can emulate
spectra with lower resolution using custom instrumental functions.

The following instrumental functions are available:
1) Rectangular
2) Triangular
3) Gaussian
4) Diffraction
5) Michelson
6) Dispersion
7) Lorentz

To get a description of each instrumental function we can use getHelp():
>>> getHelp(slit_functions)
  RECTANGULAR : SLIT_RECTANGULAR
  TRIANGULAR  : SLIT_TRIANGULAR
  GAUSSIAN    : SLIT_GAUSSIAN
  DIFFRACTION : SLIT_DIFFRACTION
  MICHELSON   : SLIT_MICHELSON
  DISPERSION/LORENTZ : SLIT_DISPERSION

For instance,
>>> getHelp(SLIT_MICHELSON)
... will give a datailed info about Michelson's instrumental function.


The function convolveSpectrum() convolutes a high-resulution spectrum
with one of supplied instrumental (slit) functions. The folowing
parameters of this function are provided:

Wavenumber     (required parameter)

  Array of wavenumbers in high-resolution input spectrum.
  Deprecated name is Omega.

CrossSection     (required parameter)

  Values of high-resolution input spectrum.

Resolution     (optional parameter)

  This parameter is passed to the slit function. It represents
  the resolution of corresponding instrument.
  NOTE: default value is 0.1 cm-1

AF_wing     (optional parameter)

  Width of an instrument function where it is considered non-zero.
  NOTE: default value is 10.0 cm-1

SlitFunction     (optional parameter)

  Custom instrumental function to convolve with spectrum.
  Format of the instrumental function must be as follows:
  Func(x, g), where Func - function name, x - wavenumber,
  g - resolution.
  NOTE: if omitted, then the default value is SLIT_RECTANGULAR


Before using the convolution procedure it worth giving some practical
advices and remarks:
1) Quality of a convolution depends on many things: quality of calculated
spectra, width of AF_wing and WavenumberRange, Resolution, WavenumberStep etc ...
Most of these factors are taken from previus stages of spectral calculation.
Right choise of all these factors is crucial for the correct computation.
2) Dispersion, Diffraction and Michelson AF's don't work well in narrow
wavenumber range because of their broad wings.
3) Generally one must consider WavenumberRange and AF_wing as wide as possible.
4) After applying a convolution, the resulting spectral range for
the lower-resolution spectra is reduced by the doubled value of AF_wing.
For this reason, try to make an initial spectral range for high-resolution
spectrum (absorption, transmittance, radiance) sufficiently broad.

The following command will calculate a lower-resolution spectra from
the CO2 transmittance, which was calculated in a previous section.
The Spectral resolution is 1 cm-1,

>>> nu_, trans_, i1, i2, slit = convolveSpectrum(nu, trans)

The outputs are:

nu_, trans_ - wavenumbers and transmittance for the resulting
              low-resolution spectrum.

i1, i2 - indexes for initial nu, trans spectrum denoting the part of
        wavenumber range which was taken for lower resolution spectrum.
        => Low-res spectrum is calculated on nu[i1:i2]

Note, than to achieve more flexibility, one have to specify most of
the optional parameters. For instance, more complete call is as follows:
>>> nu_, trans_, i1, i2, slit = convolveSpectrum(nu, trans, SlitFunction=SLIT_MICHELSON, Resolution=1.0, AF_wing=20.0)

"""
def print_spectra_tutorial():
    pydoc.pager(spectra_tutorial_text)

plotting_tutorial_text = \
"""

PLOTTING THE SPECTRA WITH MATPLOTLIB

This tutorial briefly explains how to make plots using
the Matplotlib - Python library for plotting.

Prerequisites:
   To tun through this tutorial, user must have the following
   Python libraries installed:
   1) Matplotlib
       Matplotlib can be obtained by the link http://matplotlib.org/
   2) Numpy  (required by HAPI itself)
       Numpy can be obtained via pip:
          sudo pip install numpy (under Linux and Mac)
          pip install numpy (under Windows)
       Or by the link http://www.numpy.org/

As an option, user can download one of the many scientific Python
distributions, such as Anaconda, Canopy etc...

So, let's calculate plot the basic entities which ar provided by HAPI.
To do so, we will do all necessary steps to download, filter and
calculate cross sections "from scratch". To demonstrate the different
possibilities of matplotlib, we will mostly use Pylab - a part of
Matplotlib with the interface similar to Matlab. Please note, that it's
not the only way to use Matplotlib. More information can be found on it's site.

The next part is a step-by-step guide, demonstrating basic possilities
of HITRANonline API in conjunction with Matplotlib.

First, do some preliminary imports:
>>> from hapi import *
>>> from pylab import show, plot, subplot, xlim, ylim, title, legend, xlabel, ylabel, hold

Start the database 'data':
>>> db_begin('data')

Download lines for main isotopologue of ozone in [3900, 4050] range:
>>> fetch('O3', 3, 1, 3900, 4050)

PLot a sick spectrum using the function getStickXY()
>>> x, y = getStickXY('O3')
>>> plot(x, y); show()

Zoom in spectral region [4020, 4035] cm-1:
>>> plot(x, y); xlim([4020, 4035]); show()

Calculate and plot difference between Voigt and Lorentzian lineshape:
>>> wn = arange(3002, 3008, 0.01) # get wavenumber range of interest
>>> voi = PROFILE_VOIGT(3005, 0.1, 0.3, wn)[0]   # calc Voigt
>>> lor = PROFILE_LORENTZ(3005, 0.3, wn)   # calc Lorentz
>>> diff = voi-lor    # calc difference
>>> subplot(2, 1, 1)   # upper panel
>>> plot(wn, voi, 'red', wn, lor, 'blue')  # plot both profiles
>>> legend(['Voigt', 'Lorentz'])   # show legend
>>> title('Voigt and Lorentz profiles')   # show title
>>> subplot(2, 1, 2)   # lower panel
>>> plot(wn, diff)   # plot diffenence
>>> title('Voigt-Lorentz residual')   # show title
>>> show()   # show all figures

Calculate and plot absorption coefficients for ozone using Voigt
profile. Spectra are calculated for 4 cases of thermodynamic parameters:
(1 atm, 296 K), (5 atm, 296 K), (1 atm, 500 K), and (5 atm, 500 K)
>>> nu1, coef1 = absorptionCoefficient_Voigt(((3, 1), ), 'O3',
        WavenumberStep=0.01, HITRAN_units=False, GammaL='gamma_self',
        Environment={'p':1, 'T':296.})
>>> nu2, coef2 = absorptionCoefficient_Voigt(((3, 1), ), 'O3',
        WavenumberStep=0.01, HITRAN_units=False, GammaL='gamma_self',
        Environment={'p':5, 'T':296.})
>>> nu3, coef3 = absorptionCoefficient_Voigt(((3, 1), ), 'O3',
        WavenumberStep=0.01, HITRAN_units=False, GammaL='gamma_self',
        Environment={'p':1, 'T':500.})
>>> nu4, coef4 = absorptionCoefficient_Voigt(((3, 1), ), 'O3',
        WavenumberStep=0.01, HITRAN_units=False, GammaL='gamma_self',
        Environment={'p':5, 'T':500.})
>>> subplot(2, 2, 1); plot(nu1, coef1); title('O3 k(w): p=1 atm, T=296K')
>>> subplot(2, 2, 2); plot(nu2, coef2); title('O3 k(w): p=5 atm, T=296K')
>>> subplot(2, 2, 3); plot(nu3, coef3); title('O3 k(w): p=1 atm, T=500K')
>>> subplot(2, 2, 4); plot(nu4, coef4); title('O3 k(w): p=5 atm, T=500K')
>>> show()

Calculate and plot absorption, transmittance and radiance spectra for 1 atm
and 296K. Path length is set to 10 m.
>>> nu, absorp = absorptionSpectrum(nu1, coef1, Environment={'l':1000.})
>>> nu, transm = transmittanceSpectrum(nu1, coef1, Environment={'l':1000.})
>>> nu, radian = radianceSpectrum(nu1, coef1, Environment={'l':1000., 'T':296.})
>>> subplot(2, 2, 1); plot(nu1, coef1, 'r'); title('O3 k(w): p=1 atm, T=296K')
>>> subplot(2, 2, 2); plot(nu, absorp, 'g'); title('O3 absorption: p=1 atm, T=296K')
>>> subplot(2, 2, 3); plot(nu, transm, 'b'); title('O3 transmittance: p=1 atm, T=296K')
>>> subplot(2, 2, 4); plot(nu, radian, 'y'); title('O3 radiance: p=1 atm, T=296K')
>>> show()

Calculate and compare high resolution spectrum for O3 with lower resolution
spectrum convoluted with an instrumental function of ideal Michelson interferometer.
>>> nu_, trans_, i1, i2, slit = convolveSpectrum(nu, transm, SlitFunction=SLIT_MICHELSON, Resolution=1.0, AF_wing=20.0)
>>> plot(nu, transm, 'red', nu_, trans_, 'blue'); legend(['HI-RES', 'Michelson']); show()

"""
def print_plotting_tutorial():
    pydoc.pager(plotting_tutorial_text)

def getHelp(arg=None):
    """
    This function provides interactive manuals and tutorials.
    """
    if arg == None:
        print('--------------------------------------------------------------')
        print('Hello, this is an interactive help system of HITRANonline API.')
        print('--------------------------------------------------------------')
        print('Run getHelp(.) with one of the following arguments:')
        print('    tutorial  -  interactive tutorials on HAPI')
        print('    units     -  units used in calculations')
        print('    index     -  index of available HAPI functions')
    elif arg == 'tutorial':
        print('-----------------------------------')
        print('This is a tutorial section of help.')
        print('-----------------------------------')
        print('Please choose the subject of tutorial:')
        print('    data      -  downloading the data and working with it')
        print('    spectra   -  calculating spectral functions')
        print('    plotting  -  visualizing data with matplotlib')
        print('    python    -  Python quick start guide')
    elif arg == 'python':
        print_python_tutorial()
    elif arg == 'data':
        print_data_tutorial()
    elif arg == 'spectra':
        print_spectra_tutorial()
    elif arg == 'plotting':
        print_plotting_tutorial()
    elif arg == 'index':
        print('------------------------------')
        print('FETCHING DATA:')
        print('------------------------------')
        print('  fetch')
        print('  fetch_by_ids')
        print('')
        print('------------------------------')
        print('WORKING WITH DATA:')
        print('------------------------------')
        print('  db_begin')
        print('  db_commit')
        print('  tableList')
        print('  describe')
        print('  select')
        print('  sort')
        print('  extractColumns')
        print('  getColumn')
        print('  getColumns')
        print('  dropTable')
        print('')
        print('------------------------------')
        print('CALCULATING SPECTRA:')
        print('------------------------------')
        print('  profiles')
        print('  partitionSum')
        print('  absorptionCoefficient_HT')
        print('  absorptionCoefficient_Voigt')
        print('  absorptionCoefficient_SDVoigt')
        print('  absorptionCoefficient_Lorentz')
        print('  absorptionCoefficient_Doppler')
        print('  transmittanceSpectrum')
        print('  absorptionSpectrum')
        print('  radianceSpectrum')
        print('')
        print('------------------------------')
        print('CONVOLVING SPECTRA:')
        print('------------------------------')
        print('  convolveSpectrum')
        print('  slit_functions')
        print('')
        print('------------------------------')
        print('INFO ON ISOTOPOLOGUES:')
        print('------------------------------')
        print('  ISO_ID')
        print('  abundance')
        print('  molecularMass')
        print('  moleculeName')
        print('  isotopologueName')
        print('')
        print('------------------------------')
        print('MISCELLANEOUS:')
        print('------------------------------')
        print('  getStickXY')
        print('  read_hotw')
    elif arg == ISO:
        print_iso()
    elif arg == ISO_ID:
        print_iso_id()
    elif arg == profiles:
        print_profiles()
    elif arg == slit_functions:
        print_slit_functions()
    else:
       help(arg)
