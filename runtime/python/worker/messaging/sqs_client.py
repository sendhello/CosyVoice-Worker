from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import boto3


@dataclass
class SQSMessage:
    """A lightweight container for an SQS message and its parsed payload."""

    message_id: str
    receipt_handle: str
    body_raw: str
    body: Dict[str, Any]


class SQSClient:
    """Thin wrapper over boto3 SQS client focused on receive/delete.

    This class intentionally exposes only a narrow API used by the worker to
    decouple from boto3 and streamline testing.
    """

    def __init__(
        self,
        queue_url: str,
        aws_region: Optional[str] = None,
        aws_profile: Optional[str] = None,
    ) -> None:
        self.queue_url = queue_url
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile, region_name=aws_region)
        else:
            session = boto3.Session(region_name=aws_region)
        self.client = session.client('sqs')

    def receive(
        self,
        max_messages: int = 10,
        wait_time_seconds: int = 20,
        visibility_timeout: Optional[int] = None,
    ) -> List[SQSMessage]:
        """Receive up to `max_messages` SQS messages (long polling).

        Returns a list of `SQSMessage`. If the queue is empty, the list is empty.
        """
        params = {
            'QueueUrl': self.queue_url,
            'MaxNumberOfMessages': max(1, min(max_messages, 10)),
            'WaitTimeSeconds': max(0, min(wait_time_seconds, 20)),
            'MessageAttributeNames': ['All'],
        }
        if visibility_timeout is not None:
            params['VisibilityTimeout'] = visibility_timeout
        resp = self.client.receive_message(**params)
        messages = []
        for m in resp.get('Messages', []) or []:
            body_raw = m.get('Body', '') or ''
            try:
                body = json.loads(body_raw)
                if not isinstance(body, dict):
                    body = {'value': body}
            except Exception:
                body = {'raw': body_raw}
            messages.append(
                SQSMessage(
                    message_id=m.get('MessageId', ''),
                    receipt_handle=m.get('ReceiptHandle', ''),
                    body_raw=body_raw,
                    body=body,
                )
            )
        return messages

    def delete(self, receipt_handle: str) -> None:
        """Delete a message by its receipt handle (ack)."""
        self.client.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)

    def delete_batch(self, receipt_handles: List[str]) -> None:
        """Delete a batch of messages (ack many)."""
        if not receipt_handles:
            return
        entries = [
            {'Id': str(i), 'ReceiptHandle': rh}
            for i, rh in enumerate(receipt_handles)
        ]
        self.client.delete_message_batch(QueueUrl=self.queue_url, Entries=entries)
