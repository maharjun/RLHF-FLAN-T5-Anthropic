"""
This file contains the code for some very helpful utilities to store data and
perform auxiliary actions periodically in training loops and the like.

Taken from below under the BSD 3-Clause License
    
    https://gist.github.com/maharjun/f9fdd09406c065a72e31b93d05a40f42
"""

###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2023, maharjun
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

from math import ceil

# These are some functions that help implement floating point based interval checking
def _sym_mod(n1, n2):
    mod = n1 % n2
    if mod > (n2/2):
        return mod - n2
    else:
        return mod


def _is_near_div(n1, n2):
    return -0.5 < _sym_mod(n1, n2) <= 0.5


class LoopInterval:
    """
    This class implements logic that makes it easy to do something every so often
    in a loop. Why do we need a class for this? because the logic that handles
    intervals of floating point length requires the use of a few state variables
    and is best hidden from daylight.
    """
    def __init__(self, iters_per_interval, max_n_intervals=-1, at_iter_start=False):
        """
        Parameters
        ----------

        iters_per_interval: float
            The number of iterations per interval. Can be a floating point number

        max_n_intervals: int (default -1)
            The maximum number of intervals for which self.is_interval_complete
            returns True. Useful if we want to do something only certain number of
            times in the beginning. A negative value means that the interval
            completion is triggered indefinitely

        at_iter_start: bool (default False)
            This is a flag that specifies whether this interval is considered
            complete at the point where is_interval_complete is called. This
            affects the first value returned by self.iters_in_previous_interval.
            If True, then self.iters_in_previous_interval() returns 0 the first
            time and 1 if False
        """

        if iters_per_interval < 1:
            raise ValueError("The number of iterations per interval must be at least 1")

        self._iters_per_interval = iters_per_interval
        self._max_n_intervals = max_n_intervals
        self._at_iter_start = at_iter_start

        # state variables
        self._iter_counter = 0
        self._prev_interval_iter = None
        self._curr_interval_iter = None
        self._interval_counter = 0

    def _is_interval_complete_at_iter(self, iter):
        return (_is_near_div(self._iter_counter, self._iters_per_interval)
                and (self._max_n_intervals < 0 or self._interval_counter < self._max_n_intervals))

    def is_interval_complete(self):
        """
        This function must be called exactly once per loop iteration. If the number
        of iterations has become proportional to iters_per_interval, it returns
        True. If `max_n_intervals` is specified, then it will return True for the
        first `max_n_intervals` calls and then return false.
        """

        is_interval_complete = self._is_interval_complete_at_iter(self._iter_counter)
        if is_interval_complete:
            self._prev_interval_iter = self._curr_interval_iter
            self._curr_interval_iter = self._iter_counter
            self._interval_counter += 1

        self._iter_counter += 1
        return is_interval_complete

    def iters_since_last_interval(self):
        if self._curr_interval_iter is None:
            return None
        return self._iter_counter - self._curr_interval_iter - int(not self._at_iter_start)

    def iters_in_previous_interval(self):
        if self._prev_interval_iter is None and self._curr_interval_iter is None:
            return None
        elif self._prev_interval_iter is None:
            return int(not self._at_iter_start)  # case of 0th iteration interval
        else:
            return self._curr_interval_iter - self._prev_interval_iter

    def reset(self):
        """
        Resets internal state to the way it was after construction. After calling
        reset, this can be used in another loop
        """
        self._iter_counter = 0
        self._prev_interval_iter = None
        self._curr_interval_iter = None
        self._interval_counter = 0


class NeverLoopInterval(LoopInterval):

    def __init__(self):
        pass

    def is_interval_complete(self):
        return False

    def iters_in_previous_interval(self):
        raise NotImplementedError("NeverLoopInterval does not implement iters_in_previous_interval")

    def reset(self):
        pass


def get_loop_interval(n_points_per_epoch, batch_size,
                      interval_val=1, interval_type='epochs', max_n_intervals=-1, split_across_epochs=True):
    """
    Get a LoopInterval object for a typical training loop where data is fed in
    batches over epochs. The assumption here is that each iteration of the loop
    deals with one batch

    Parameters
    ----------

    n_points_per_epoch: int
        The number of data points in one epoch

    batch_size: int
        The number of data points in each batch

    interval_val: int
        An integer that is interpreted differently according to the specified interval_type

    interval_type: str
        one of three values

        - 'epochs': In this case an interval is triggered once every `interval_val` epochs
        - 'per-epoch': In this case an interval is triggered `interval_val` times per epoch
        - 'batches': In this case an interval is triggered every `interval_val` batches

    max_n_intervals: int (default: -1)
        The maximum number of intervals triggered. If non-negative, a maximum of
        `max_n_intervals` will be triggered. If negative, the intervals are
        triggered indefinitely

    split_across_epochs: bool (default: True)
        This conveys to the interval counter the behaviour of batches across epoch
        boundaries. If batches are split across epoch boundaries (as is typical
        with pytorch data loaders, which give the final batch using the remaining
        elements of the epoch), this value should be True. If on the other hand,
        the batches stretch across epoch boundaries while maintaining a constant
        size (using a custom batcher for instance), set this to False.

    Returns
    -------

    A LoopInterval object for which the `is_interval_complete` method returns true
    as often as specified by the interval parameters specified here
    """
    n_points_per_epoch = int(n_points_per_epoch)
    batch_size = int(batch_size)

    if split_across_epochs:
        n_batches_per_epoch = ceil(n_points_per_epoch / batch_size)
    else:
        n_batches_per_epoch = float(n_points_per_epoch / batch_size)

    valid_interval_types = {'per-epoch', 'epochs', 'batches', 'never'}
    if interval_type not in valid_interval_types:
        raise ValueError(f"Unable to understand interval type '{interval_type}', should be one of '{valid_interval_types}'")

    if interval_type == 'never':
        return NeverLoopInterval()

    if interval_type == 'per-epoch':
        batches_per_interval = n_batches_per_epoch / interval_val
    elif interval_type == 'epochs':
        batches_per_interval = n_batches_per_epoch * interval_val
    elif interval_type == 'batches':
        batches_per_interval = interval_val        

    return LoopInterval(batches_per_interval, max_n_intervals)
