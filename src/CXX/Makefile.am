# Compiler & linker flags
AM_CXXFLAGS = -g -Wall -Werror
AM_LDFLAGS  =

# Enable debug messages
if ENABLE_DEBUG
AM_CXXFLAGS += -DENABLE_DEBUG
endif


SUBDIRS = \
    libnn \
    unit_test


# Internal headers
noinst_HEADERS = \
    config.hxx

#
# No shared object is necessary so far
# Uncomment in case it becomes so
#lib_LTLIBRARIES = \
#    libnn.la
#
#libnn_la_SOURCES =
#
#libnn_la_LIBADD =
#
#LDADD = libnn.la


# Executables (none so far)
#bin_PROGRAMS = \
#    nn_something
#
#nn_something_SOURCES = \
#    something.cxx
