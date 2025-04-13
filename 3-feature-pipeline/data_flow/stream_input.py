import json
import base64
from datetime import datetime
import time
from typing import Generic, Iterable, List, Optional, TypeVar, Dict, Any

from bytewax.inputs import FixedPartitionedSource, StatefulSourcePartition
from config import settings
from mq import RabbitMQConnection
from utils.logging import get_logger

logger = get_logger(__name__)

DataT = TypeVar("DataT")
MessageT = TypeVar("MessageT")


class RabbitMQPartition(StatefulSourcePartition, Generic[DataT, MessageT]):
    """
    Class responsible for creating a connection between bytewax and rabbitmq that facilitates the transfer of data from mq to bytewax streaming pipeline.
    Inherits StatefulSourcePartition for snapshot functionality that enables saving the state of the queue
    """

    def __init__(self, queue_name: str, resume_state: MessageT | None = None) -> None:
        self._in_flight_msg_ids = resume_state or set()
        self.queue_name = queue_name
        self.connection = RabbitMQConnection()
        self.connection.connect()
        self.channel = self.connection.get_channel()

    def next_batch(self, sched: Optional[datetime]) -> Iterable[DataT]:
        try:
            method_frame, header_frame, body = self.channel.basic_get(
                queue=self.queue_name, auto_ack=True
            )
        except Exception:
            logger.error(
                f"Error while fetching message from queue.", queue_name=self.queue_name
            )
            time.sleep(10)  # Sleep for 10 seconds before retrying to access the queue.

            self.connection.connect()
            self.channel = self.connection.get_channel()

            return []

        if method_frame:
            message_id = method_frame.delivery_tag
            self._in_flight_msg_ids.add(message_id)

            # Parse the message body
            try:
                message = json.loads(body)
                
                # Handle binary data if present
                if message.get("type") == "image_documents" and "image_data_base64" in message:
                    # Convert base64 image data to binary
                    try:
                        base64_data = message.pop("image_data_base64")
                        message["image_data"] = base64.b64decode(base64_data)
                        logger.info(f"Decoded binary image data from base64, size: {len(message['image_data'])} bytes")
                    except Exception as e:
                        logger.error(f"Error decoding image data: {str(e)}")
                        message["image_data"] = b""  # Empty binary data as fallback
                
                return [message]
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON message received: {body[:200]}...")
                return []
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                return []
        else:
            return []

    def snapshot(self) -> MessageT:
        return self._in_flight_msg_ids

    def garbage_collect(self, state):
        closed_in_flight_msg_ids = state
        for msg_id in closed_in_flight_msg_ids:
            self.channel.basic_ack(delivery_tag=msg_id)
            self._in_flight_msg_ids.remove(msg_id)

    def close(self):
        self.channel.close()


class RabbitMQSource(FixedPartitionedSource):
    def list_parts(self) -> List[str]:
        return ["single partition"]

    def build_part(
        self, now: datetime, for_part: str, resume_state: MessageT | None = None
    ) -> StatefulSourcePartition[DataT, MessageT]:
        return RabbitMQPartition(queue_name=settings.RABBITMQ_QUEUE_NAME)