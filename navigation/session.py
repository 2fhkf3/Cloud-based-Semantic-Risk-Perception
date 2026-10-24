"""Mission selection is an explicit lifecycle event, not a property of a late packet."""
from .schema import identifier


class SessionGuard:
    def __init__(self):
        self.current = None
        self.retired = set()

    def start(self, session_id):
        identifier(session_id, 'session_id')
        if session_id == self.current:
            return False
        if session_id in self.retired:
            raise ValueError('a retired session cannot be restarted; use a new mission ID')
        if self.current is not None:
            self.retired.add(self.current)
        self.current = session_id
        return True

    def accepts(self, session_id):
        # Permit bootstrap only. Later changes require a trusted start_session event.
        if self.current is None:
            self.start(session_id)
        return self.current == session_id
