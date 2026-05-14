"""Session-level conversation logging for IDE agents.

Tracks what agents learn, prevents looping, provides context awareness.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversationEntry:
    """Single conversation turn (user prompt + agent response)."""
    id: str
    timestamp: str
    user_prompt: str
    agent_action: str  # What the agent did (e.g., "Query codebase", "Update metadata")
    discovered_patterns: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    metadata_updated: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    reason: str = ""  # Why this action was taken


@dataclass
class SessionLog:
    """Complete IDE session log."""
    project: str
    start_time: str
    conversations: List[ConversationEntry] = field(default_factory=list)
    learned_constraints: List[str] = field(default_factory=list)
    learned_architecture: List[str] = field(default_factory=list)
    
    def add_conversation(self, entry: ConversationEntry) -> None:
        """Add a new conversation entry."""
        self.conversations.append(entry)
        
        # Auto-extract patterns to learned lists
        for pattern in entry.discovered_patterns:
            if pattern not in self.learned_constraints:
                self.learned_constraints.append(pattern)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "project": self.project,
            "start_time": self.start_time,
            "conversations": [asdict(c) for c in self.conversations],
            "learned_constraints": self.learned_constraints,
            "learned_architecture": self.learned_architecture,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SessionLog:
        """Load from dict."""
        log = SessionLog(
            project=data.get("project", "unknown"),
            start_time=data.get("start_time", datetime.now(timezone.utc).isoformat()),
        )
        for conv_data in data.get("conversations", []):
            entry = ConversationEntry(
                id=conv_data.get("id", ""),
                timestamp=conv_data.get("timestamp", ""),
                user_prompt=conv_data.get("user_prompt", ""),
                agent_action=conv_data.get("agent_action", ""),
                discovered_patterns=conv_data.get("discovered_patterns", []),
                files_modified=conv_data.get("files_modified", []),
                metadata_updated=conv_data.get("metadata_updated", {}),
                summary=conv_data.get("summary", ""),
                reason=conv_data.get("reason", ""),
            )
            log.conversations.append(entry)
        log.learned_constraints = data.get("learned_constraints", [])
        log.learned_architecture = data.get("learned_architecture", [])
        return log


class ConversationLogger:
    """Manages conversation logging for IDE sessions."""
    
    def __init__(self, project_root: Path):
        """Initialize logger for a project.
        
        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self.session_dir = project_root / ".skeletongraph" / "session"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.session_dir / "conversation_log.json"
        self.log = self._load_or_create_log()
    
    def _load_or_create_log(self) -> SessionLog:
        """Load existing log or create new one."""
        if self.log_file.exists():
            try:
                data = json.loads(self.log_file.read_text())
                return SessionLog.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load session log: {e}, creating new")
        
        return SessionLog(
            project=self.project_root.name,
            start_time=datetime.now(timezone.utc).isoformat(),
        )
    
    def save(self) -> None:
        """Persist log to disk."""
        try:
            self.log_file.write_text(
                json.dumps(self.log.to_dict(), indent=2)
            )
        except Exception as e:
            logger.error(f"Failed to save session log: {e}")
    
    def add_entry(
        self,
        user_prompt: str,
        agent_action: str,
        discovered_patterns: Optional[List[str]] = None,
        files_modified: Optional[List[str]] = None,
        metadata_updated: Optional[Dict[str, Any]] = None,
        summary: str = "",
        reason: str = "",
    ) -> ConversationEntry:
        """Add a conversation entry.
        
        Args:
            user_prompt: What the user asked
            agent_action: What the agent did
            discovered_patterns: New patterns/constraints discovered
            files_modified: Files changed
            metadata_updated: Metadata changes (project.md, architecture.md)
            summary: Summary of the action
            reason: Why this action was taken
        
        Returns:
            The created entry
        """
        entry = ConversationEntry(
            id=f"conv_{len(self.log.conversations) + 1}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_prompt=user_prompt,
            agent_action=agent_action,
            discovered_patterns=discovered_patterns or [],
            files_modified=files_modified or [],
            metadata_updated=metadata_updated or {},
            summary=summary,
            reason=reason,
        )
        self.log.add_conversation(entry)
        self.save()
        return entry
    
    def get_context_for_agent(self) -> str:
        """Get formatted context for agent from session history.
        
        Prevents agent from re-learning or looping.
        """
        if not self.log.conversations:
            return ""
        
        context_lines = [
            "## Session Context (What We've Already Done)",
            f"Project: {self.log.project}",
            f"Learned Constraints: {', '.join(self.log.learned_constraints[:5]) or 'None yet'}",
            "",
            "Recent Actions:",
        ]
        
        for entry in self.log.conversations[-3:]:  # Last 3 conversations
            context_lines.append(f"- {entry.timestamp}: {entry.summary}")
            if entry.files_modified:
                context_lines.append(f"  Files: {', '.join(entry.files_modified)}")
        
        context_lines.extend([
            "",
            "To avoid loops: Check learned constraints above before asking.",
            "If you discover NEW constraints/patterns, add them to project.md.",
        ])
        
        return "\n".join(context_lines)
    
    def was_recently_queried(self, query: str, minutes_back: int = 10) -> bool:
        """Check if similar query was made recently.
        
        Prevents agent from repeatedly querying the same thing.
        """
        if not self.log.conversations:
            return False
        
        cutoff = datetime.now(timezone.utc).timestamp() - (minutes_back * 60)
        query_lower = query.lower()
        
        for entry in reversed(self.log.conversations):
            try:
                ts = datetime.fromisoformat(entry.timestamp).timestamp()
                if ts < cutoff:
                    break
                
                if query_lower in entry.user_prompt.lower():
                    return True
            except Exception:
                continue
        
        return False
    
    def get_last_queried(self, query_type: str) -> Optional[ConversationEntry]:
        """Get last conversation of a certain type."""
        query_lower = query_type.lower()
        
        for entry in reversed(self.log.conversations):
            if query_lower in entry.agent_action.lower():
                return entry
        
        return None


def get_conversation_logger(project_root: Path) -> ConversationLogger:
    """Get or create conversation logger for a project."""
    return ConversationLogger(project_root)
