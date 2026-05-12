import heapq
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st


class Event(object):
    """Class representing an event in the simulation"""
    ENTITY_ARRIVAL = 0
    ENTITY_COMPLETE = 1

    EVENT_TO_STR = {
        0: "ENTITY_ARRIVAL",
        1: "ENTITY_COMPLETE"
    }

    def __init__(self, event_type, time, entity, resource_id=None):
        self.event_type = event_type
        self.time = time
        self.entity = entity
        self.resource_id = resource_id  # Which resource is processing the entity

    def __lt__(self, other):
        """Less than method determines how a list or heapq of Event classes will be sorted"""
        if self.time == other.time:
            # If events have the same time, ensure FCFS by comparing entity IDs
            # This assumes entities with lower IDs arrive first
            return self.entity.id < other.entity.id
        return self.time < other.time

    def __str__(self) -> str:
        """String method determines how an Event class will be printed"""
        resource_info = f" on Resource {self.resource_id}" if self.resource_id is not None else ""
        return f"Event {self.EVENT_TO_STR[self.event_type]} for Entity {self.entity.id}{resource_info} at time {self.time:.2f}"


class Resource(object):
    """Class representing a resource in the simulation"""
    def __init__(self, resource_id, capacity):
        self.id = resource_id
        self.capacity = capacity    # Number of entities that can be processed simultaneously
        self.current_entities = []  # List of currently processing entities
        self.queue = []             # Queue for this resource

    def seize(self, entity):
        """Assign an entity to this resource and seize the resource"""
        self.current_entities.append(entity)
        return self.id

    def release(self, entity_id):
        """Release a specific entity from the resource"""
        # Keep all entities except the one with the specified/released entity_id
        self.current_entities = [entity for entity in self.current_entities if entity.id != entity_id]

    def is_available(self):
        """Check if resource is available to process an entity"""
        return len(self.current_entities) < self.capacity


class Entity(object):
    """Class representing an entity in the simulation"""
    def __init__(self, entity_id: int, processing_time: float = None):
        self.id = entity_id
        self.processing_time = processing_time
        self.arrival_time = None
        self.departure_time = None
        self.start_times = {} # Dictionary to store start times for each machine
        self.end_times = {}     # Dictionary to store end times for each machine
        self.current_machine = None  # Which resource processed this entity


class Simulation(object):
    """Class representing the simulation"""
    def __init__(self, num_machines, machine_capacities=None):
        self.now = 0
        self.events = []
        self.machines = [Resource(i, capacity=machine_capacities[i]) for i in range(num_machines)]
        self.queue = []
        self.debug = False
        self.completed_jobs = []
        self.job_counters = 0   # Track the number of jobs generated
        self.flow_times = [] # Track waiting times for statistics

        # Define the process flow - which machines follows which
        self.machine_sequence = {
            0: 1,       # After machine 0, go to machine 1
            1: None     # After machine 1, exit the production line
        }

    def add_event(self, event) -> None:
        """Add event to future events"""
        assert event.time >= self.now

        if self.debug:
            print(f">> Schedule {event}")
        heapq.heappush(self.events, event)

    def get_next_event(self) -> Event:
        """Get next event and remove it from the heapq.
            Return None if there are no future events"""
        return heapq.heappop(self.events) if self.events else None

    def get_available_machine(self):
        """Return the first available machine, or None if all are busy"""
        for machine in self.machines:
            if machine.is_available():
                return machine
        return None

    def get_inter_arrival_time(self) -> float:
        """Return the inter-arrival time with respect to a specific distribution"""
        return random.expovariate(2)

    def get_processing_time(self, machine_id) -> float:
        """Return the processing time with respect to a specific distribution"""
        if machine_id == 0:
            return 1/3
        elif machine_id == 1:
            return random.uniform(0.0,4/5) # t_e = 2/5 is the mean value => range = [0,4/5]
        return 0

    def print_future_events(self) -> None:
        print("Future Events Queue:")

        # Sort without modifying the heap
        for event in sorted(self.events, key=lambda e: e.time):
            print(event)

    def setup(self) -> None:
        """Setup the simulation with initial events"""
        # Schedule the first arrival event
        self.add_event(Event(Event.ENTITY_ARRIVAL, self.get_inter_arrival_time(), Entity(self.job_counters)))

        if self.debug:
            print(f"Job {self.job_counters} arrives at {self.now:.2f}")

    def process_arrival_event(self, event: Event) -> None:
        """Here, we process arrival events"""
        assert event.event_type == Event.ENTITY_ARRIVAL

        job = event.entity
        job.arrival_time = self.now

        # Add to the first machine queue
        first_machine = 0
        self.machines[first_machine].queue.append(job)

        if self.debug:
            print(f"Job {job.id} added to waiting queue at {self.now:.2f}")

        # Schedule the next arrival job
        self.job_counters += 1
        self.add_event(Event(Event.ENTITY_ARRIVAL, self.now + self.get_inter_arrival_time(), Entity(self.job_counters)))

        # Check if machines are available to process this job
        self.request_machines(first_machine)

    def process_departure_event(self, event: Event) -> None:
        """Here, we process departure events (job completion)"""
        assert event.event_type == Event.ENTITY_COMPLETE

        job = event.entity
        machine_id = event.resource_id

        # Record the end time in the current machine
        job.end_times[machine_id] = self.now

        if self.debug:
            print(f"Job {job.id} finishes processing on Machine {machine_id} at {self.now:.2f}")

        # Release the current machine
        self.machines[machine_id].release(job.id)

        # Check if we can process more jobs in the current machine
        self.request_machines(machine_id)

        # Determine the next machine in the sequence
        next_machine_id = self.machine_sequence[machine_id]

        if next_machine_id is None:
            # Calculate and record flow time
            job.departure_time = job.end_times[machine_id]
            self.flow_times.append(job.departure_time - job.arrival_time)

            # Collect job completed
            self.completed_jobs.append(job)
        else:
            # Move to the queue of next machine
            self.machines[next_machine_id].queue.append(job)

            # Check if we can process more jobs in the next machine
            self.request_machines(next_machine_id)

    def request_machines(self, machine_id) -> None:
        """Request machines to process jobs in the waiting queue"""
        # Define specific machine on which jobs are processed
        machine = self.machines[machine_id]

        # Continue as long as we have jobs waiting and machines available
        while machine.queue and machine.is_available():
            # Get next job from the queue (FCFS)
            job = machine.queue.pop(0)

            # Assign job to machine and seize the machine
            machine.seize(job)
            job.start_times[machine_id] = self.now
            job.current_machine = machine_id

            if self.debug:
                print(f"Job {job.id} starts processing on Machine {machine_id} at {self.now:.2f}")

            # Schedule job completion
            self.add_event(Event(Event.ENTITY_COMPLETE, self.now + self.get_processing_time(machine_id), job, resource_id=machine_id))

    def run(self, debug_flag, num_jobs):
        """Run simulation"""
        # Initialize parameters and events
        self.debug = debug_flag
        self.setup()

        # Main loop of the simulation
        event_handlers = {
            Event.ENTITY_ARRIVAL: self.process_arrival_event,
            Event.ENTITY_COMPLETE: self.process_departure_event,
        }

        while self.events and len(self.completed_jobs) < num_jobs:
            # Get the next event in the queue
            event = self.get_next_event()
            if not event:
                break

            self.now = event.time

            # Process events
            if self.debug:
                print(f"Execute {event}")

            # Call the appropriate handler based on event type
            event_handlers[event.event_type](event)

        # Return statistics
        return np.mean(self.flow_times)


if __name__ == "__main__":
    """Setup parameters and run the simulation"""
    # Model parameters
    num_jobs = pow(10,5)     # Number of jobs to simulate
    num_machines = 2         # Number of individual machines (using Resource class)
    machine_capacities = [1,1]

    # Debug enabled
    debug_on = False

    max_n_sim = 10
    results = []

    for idx_sim in range(0, max_n_sim):
        print(f"Simulation {idx_sim} out of {max_n_sim}")
        # Setup the simulation
        sim = Simulation(num_machines, machine_capacities)

        # Run the simulation
        avg_flow_time = sim.run(debug_on, num_jobs)

        results.append(avg_flow_time)

    mean_flow_time = np.mean([flow_time for flow_time in results])
    min_flow_time = min([flow_time for flow_time in results])
    max_flow_time = max([flow_time for flow_time in results])

    print(f"\nThe mean flow time is {mean_flow_time:.2f} minutes.")
    print(f"[Min, Max] = [{min_flow_time:.2f}, {max_flow_time:.2f}] minutes.")