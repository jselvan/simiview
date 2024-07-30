from typing import Generator

from simiview.simiview.trial import Trial

def filter_trials(trials, conditions, attributes) -> Generator[Trial]:
    for trial in trials:
        if not trial.condition in conditions:
            continue
        for attribute, values in attributes.items():
            if not trial.attributes.get(attribute) in values:
                break
        else:
            yield trial

class App:
    def get_events(self, event: str, index_filter: None | list[int]=None):
        trials = filter_trials(self.trials, self.filters['beh.conditions'], self.filters['beh.attributes'])
        events = []
        for trial in trials:
            timestamps = trial.marker_dict.get(event, [])
            for idx, timestamp in enumerate(timestamps):
                if index_filter is not None and idx not in index_filter:
                    continue
                events.append({
                    'trialid': trial.trialid, 
                    'timestamp': timestamp, 
                    'relative': trial.relative_to(timestamp)
                })
        for idx, event in enumerate(events):
            event['eventid'] = idx
        return events
