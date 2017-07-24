import tensorflow as tf
import pickle
from threading import Thread, Lock, Event
import Queue
from util.log import Logger

log = Logger('messaging', 'Messaging')

class ClusterMessagingClient(object):
    '''
    Allows RPC calls between cluster nodes.
    To expose RPC methods, one should derive from this class and add apropos methods to it.
    '''
    def __init__(self, cluster_spec, job, task, queue_size=1000):
        '''
        Constructs a cluster node's messaging end point.
        'cluster_spec' - A ClusterSpec instance of the cluster
        'job' - Job name of the local cluster node
        'task' - Task index of the local cluster node
        'queue_size' - Maximum number of messages in cross cluster queues
        '''
        self.cluster_spec = cluster_spec
        self.job = job
        self.task = task
        self._id_pattern = 'CMCQueue-%s-%d'
        # global ID for 'shared_name' argument in queues (to connect queues across cluster)
        self.id = self._id_pattern % (job, task)
        # messages a pickled to strings - so the queues just transport one string per message
        self._ph_content = tf.placeholder(tf.string, [])
        # outbound queues for sending messages (outbound calls and responses to inbound calls)
        self._outbound_enqueues = {}
        for j in cluster_spec.jobs:
            for t in cluster_spec.task_indices(j):
                id = self._id_pattern % (j, t)
                # skip local inbound queue
                if id != self.id:
                    queue = tf.FIFOQueue(queue_size, [tf.string], name=id, shared_name=id)
                    self._outbound_enqueues[id] = queue.enqueue([self._ph_content])
        # inbound queue for receiving messages (inbound calls and responses of outbound calls)
        inbound_queue = tf.FIFOQueue(queue_size, [tf.string], name=self.id, shared_name=self.id)
        self._inbound_dequeue = inbound_queue.dequeue()
        self._inbound_close = inbound_queue.close(cancel_pending_enqueues=True)
        # response lookup for outbound RPC calls (that are waiting for a response)
        self._waiting_for_response = {}
        # response message id counter and lock
        self._response_id_counter = 0
        self._response_id_lock = Lock()
        # local-only outbound queue for decoupling (otherwise blocking) send/receive dependencies
        self._to_send = Queue.Queue()

    def _call(self, function, args):
        '''
        Internal function to call a local class method as RPC function.
        'function' - Name of the local function (member of this class)
        'args' - Tuple of call arguments
        '''
        try:
            fun = getattr(self, function, *args)
        except AttributeError:
            log.warn('Function not found: %s, arguments: %r' % (function, args))
            return
        return fun(*args)

    def _send_loop(self, session, coord):
        '''
        Performs outbound messaging loop.
        'session' - The current session
        'coord' - A thread coordinator to automatically stop the loop
        '''
        while not coord.should_stop():
            # get enqueue and feed_dict from local sending queue
            enqueue, feed_dict = self._to_send.get()
            try:
                content = session.run(enqueue, feed_dict=feed_dict)
            except (tf.errors.CancelledError, tf.errors.OutOfRangeError):
                # queue got closed
                return

    def _receive(self, session, caller, is_response, response_id, function, items):
        '''
        Executes an inbound message. Either by calling a local RPC member function or by
        continuing a halted/callback-ed local call that is waiting for a response.
        'session' - The current session
        'caller' - Id of the caller (used to respond in case of inbound RPC call)
        'is_response' - If the message is a response (and not an inbound RPC call)
        'response_id' - Response ID to either lookup waiting callbacks (response case) or
            to send back a response to the caller (in case of an inbound RPC call)
        'function' - Name of the RPC function that should be called (inbound RPC) or that
            was called to produce this response (outbound RPC)
        'items' - Arguments of inbound RPC call or result of outbound RPC call
        '''
        if is_response:
            # we got the response for an outbound RPC call.
            log.step('Response from: %s, function: %s, return values: %r' % (caller, function, items))
            callback = self._waiting_for_response[response_id]
            if callable(callback):
                # Local response is a callable - so we call it.
                callback(items)
                # Remove the local response entry - we are done here.
                del self._waiting_for_response[response_id]
            else:
                # Local response entry is an Event class that waits for a set() call
                # to continue the waiting local function call (synchronous RPC calls).
                # First we replace the Event by the callback argument/result to allow
                # the continued call to return it...
                self._waiting_for_response[response_id] = items
                # Then we call set() to let it continue.
                # Contract: 'call' function has to remove the self._waiting_for_response entry by itself.
                callback.set()
        else:
            # We got an inbound RPC call.
            log.step('Call from: %s, function: %s, arguments: %r' % (caller, function, items))
            # Calling the local RPC function/class member (if available).
            results = self._call(function, items)
            if response_id > 0:
                # We are requested to produce a response.
                # en-pickle header and results
                content = pickle.dumps((self.id, True, response_id, function, results))
                # get enqueue op of outbound local queue to the calling client
                enqueue = self._outbound_enqueues[caller]
                # locally enqueue op and feed_dict for decoupled sending
                self._to_send.put((enqueue, { self._ph_content: content }))

    def _receive_loop(self, session, coord):
        '''
        Performs inbound messaging loop.
        'session' - The current session
        'coord' - A thread coordinator to automatically stop the loop
        '''
        while not coord.should_stop():
            try:
                content = session.run(self._inbound_dequeue)
            except (tf.errors.CancelledError, tf.errors.OutOfRangeError):
                # queue got closed
                return
            # de-pickle header and call arguments
            args = (session,) + pickle.loads(content)
            # Create a thread to decouple blocking cross-cluster call/response chains.
            Thread(target=self._receive, args=args).start()

    def start_queue_threads(self, session, coord):
        '''
        Starts required threads for receiving and sending messages.
        'session' - The current session
        'coord' - A thread coordinator to automatically stop the threads
        '''
        threads = []
        for routine in [self._send_loop, self._receive_loop]:
            thread = Thread(target=routine, args=(session,coord))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        log.debug('Started queue threads')
        return threads

    def close_queues(self, session):
        '''
        Closes all queues of this instance.
        'session' - The current session
        '''
        session.run(self._inbound_close)
        log.debug('Closed inbound queue')

    def call_async(self, job, task, function, callback, *args):
        '''
        Performs an asynchronous (non-blocking) RPC call to a remote node's function.
        If 'job' and 'task' address the local node, the call is performed locally.
        'job' - Job name of the local cluster node
        'task' - Task index of the local cluster node
        'function' - Name of the remote function (member of class derived from ClusterMessagingClient)
        'callback' - Will be called with return value tuple of remote function as arguments,
            if RPC call succeeded. Omitting it (None) results in dependency free state
            (the local client can safely be stopped/ended).
            Can also be an Event object (used internally by 'call').
        'args' - Arguments to the remote function.
        Returns a callback id (0 in case of no callback).
        '''
        # target client id (shared_name for the apropos queue)
        id = self._id_pattern % (job, task)
        if self.id == id:
            # local call - no queueing needed
            log.step('Local call of function: %s, arguments: %r' % (function, args))
            results = self._call(function, args)
            if callback:
                callback(results)
            return -1
        log.step('Calling worker: %s, function: %s, arguments: %r' % (id, function, args))
        # default value in case of no callback, if not waiting for responseno
        # and no response should be produced on the receiving end.
        response_id = 0
        if callback:
            with self._response_id_lock:
                # acquire a new response id
                response_id = self._response_id_counter = self._response_id_counter + 1
                # register callback or Event object as 'waiting for response'
                self._waiting_for_response[response_id] = callback
        # en-pickle header and call arguments
        content = pickle.dumps((self.id, False, response_id, function, args))
        # get enqueue op of outbound queue of the targeted client
        enqueue = self._outbound_enqueues[id]
        # enqueue locally for sending
        self._to_send.put((enqueue, { self._ph_content: content }))
        # passing back response_id (used for synchronous RPCs - see 'call')
        return response_id

    def call(self, job, task, function, *args):
        '''
        Does a synchronous (blocking) RPC call to a remote node's function.
        If 'job' and 'task' address the local client, the call is performed locally.
        'job' - Job name of the local cluster node
        'task' - Task index of the local cluster node
        'function' - Name of the remote function (member of class derived from ClusterMessagingClient)
        'args' - Arguments to the remote function.
        '''
        if job == self.job and task == self.task:
            # local call - no queueing needed
            log.step('Local call of function: %s, arguments: %r' % (function, args))
            return self._call(function, args)
        # Create an Event object to wait/block until response arrives.
        event = Event()
        # Using async variant with Event object as callback (see '_receive').
        response_id = self.call_async(job, task, function, event, *args)
        # Waiting for '_receive' function to 'set()' it.
        event.wait()
        # Contract: getting return value from response registry
        result = self._waiting_for_response[response_id]
        # Contract: removing the entry (independent of other responses and atomar -> thread safe)
        del self._waiting_for_response[response_id]
        # passing back result of the blocking call
        return result
