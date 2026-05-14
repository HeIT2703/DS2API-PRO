import asyncio
import contextlib
import threading
from typing import AsyncGenerator, Optional, Union

from .client import DeepSeekClient
from .src.http_client import DEFAULT_TIMEOUT, Timeout
from .src.models import ChatResponse, StreamEvent

class AsyncDeepSeekClient:
    """
    Async wrapper for the unofficial chat.deepseek.com web client.
    
    This is an asynchronous wrapper around the synchronous DeepSeekClient,
    allowing it to be used seamlessly in asyncio applications without blocking the event loop.
    It uses asyncio.to_thread to execute synchronous operations.
    """
    
    def __init__(self, token: Optional[str] = None, timeout: Timeout = DEFAULT_TIMEOUT, **client_kwargs):
        self._sync_client = DeepSeekClient(token=token, timeout=timeout, **client_kwargs)

    async def new_chat(self, model: str = "instant") -> str:
        """Create a managed chat session and return its session ID."""
        return await asyncio.to_thread(self._sync_client.new_chat, model)

    async def adopt_chat(
        self,
        session_id: str,
        model: str = "instant",
        last_message_id: Optional[Union[int, str]] = None,
    ) -> str:
        """Register an existing chat session for high-level conversation methods."""
        return await asyncio.to_thread(self._sync_client.adopt_chat, session_id, model, last_message_id)

    async def delete_chat(self, session_id: str) -> dict:
        """Delete a chat session and clear local conversation state."""
        return await asyncio.to_thread(self._sync_client.delete_chat, session_id)

    async def ask(
        self,
        prompt: str,
        model: str = "instant",
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[list] = None,
        print_to_stdout: bool = False,
    ) -> ChatResponse:
        """Send a one-shot message and return the full response object."""
        return await asyncio.to_thread(
            self._sync_client.ask,
            prompt=prompt,
            model=model,
            thinking=thinking,
            search=search,
            ref_file_ids=ref_file_ids,
            print_to_stdout=print_to_stdout
        )

    async def send(
        self,
        session_id: str,
        prompt: str,
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[list] = None,
        parent_message_id: Optional[Union[int, str]] = None,
        print_to_stdout: bool = False,
    ) -> ChatResponse:
        """Send a message in an existing session and return the full ChatResponse."""
        return await asyncio.to_thread(
            self._sync_client.send,
            session_id=session_id,
            prompt=prompt,
            thinking=thinking,
            search=search,
            ref_file_ids=ref_file_ids,
            parent_message_id=parent_message_id,
            print_to_stdout=print_to_stdout
        )

    async def regenerate(self, session_id: str, message_id: Union[int, str], print_to_stdout: bool = False) -> ChatResponse:
        """Regenerate a response."""
        return await asyncio.to_thread(self._sync_client.regenerate, session_id, message_id, print_to_stdout)

    async def edit_message(self, session_id: str, message_id: Union[int, str], prompt: str, print_to_stdout: bool = False) -> ChatResponse:
        """Edit a message and return the new response."""
        return await asyncio.to_thread(self._sync_client.edit_message, session_id, message_id, prompt, print_to_stdout)

    async def continue_response(self, session_id: str, message_id: Union[int, str], print_to_stdout: bool = False) -> ChatResponse:
        """Continue a truncated response."""
        return await asyncio.to_thread(self._sync_client.continue_response, session_id, message_id, print_to_stdout)

    async def stop(self, session_id: str) -> dict:
        """Stop an active generation stream in a session."""
        return await asyncio.to_thread(self._sync_client.stop, session_id)

    async def _async_generator_wrapper(self, sync_generator_func, *args, **kwargs) -> AsyncGenerator[StreamEvent, None]:
        """Wrap a synchronous generator into an asynchronous one using a queue."""
        queue = asyncio.Queue()
        sentinel = object()
        loop = asyncio.get_running_loop()
        stop_requested = threading.Event()
        sync_generator = None

        def _run_sync():
            nonlocal sync_generator
            try:
                sync_generator = sync_generator_func(*args, **kwargs)
                for item in sync_generator:
                    if stop_requested.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except Exception as e:
                if not stop_requested.is_set():
                    loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                if not loop.is_closed():
                    loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        # Run the synchronous generator in a separate thread
        thread_task = asyncio.create_task(asyncio.to_thread(_run_sync))
        consumed_to_end = False

        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    consumed_to_end = True
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
                queue.task_done()
        finally:
            if not consumed_to_end:
                stop_requested.set()
                if sync_generator is not None and hasattr(sync_generator, "close"):
                    with contextlib.suppress(Exception):
                        sync_generator.close()

            if consumed_to_end:
                await thread_task
            elif not thread_task.done():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(thread_task), timeout=1.0)

    def ask_stream(
        self,
        prompt: str,
        model: str = "instant",
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[list] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Send a one-shot message and stream the response events."""
        return self._async_generator_wrapper(
            self._sync_client.ask_stream,
            prompt=prompt,
            model=model,
            thinking=thinking,
            search=search,
            ref_file_ids=ref_file_ids,
        )

    def send_stream(
        self,
        session_id: str,
        prompt: str,
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[list] = None,
        parent_message_id: Optional[Union[int, str]] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Send a message in an existing session and stream response events."""
        return self._async_generator_wrapper(
            self._sync_client.send_stream,
            session_id=session_id,
            prompt=prompt,
            thinking=thinking,
            search=search,
            ref_file_ids=ref_file_ids,
            parent_message_id=parent_message_id,
        )

    async def close(self):
        """Close the underlying HTTP session."""
        await asyncio.to_thread(self._sync_client.close)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
