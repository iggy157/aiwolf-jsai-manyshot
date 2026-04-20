"""Module defining a stoppable thread class.

スレッドを停止できるようにするためのクラスを定義するモジュール.
"""

import ctypes
import threading


class StoppableThread(threading.Thread):
    """A thread class that can be stopped gracefully.

    スレッドを停止できるようにするためのクラス.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[arg-type]  # noqa: ANN002, ANN003
        """Initialize a stoppable thread.

        スレッドを停止できるようにするためのクラスを初期化する.

        Args:
            *args: Variable length argument list for threading.Thread / threading.Thread用の可変長引数リスト
            **kwargs: Arbitrary keyword arguments for threading.Thread / threading.Thread用の任意のキーワード引数
        """
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Stop the thread execution.

        スレッドを停止するためのメソッド.
        """
        if not self.is_alive():
            return

        # スレッドIDを取得して強制終了
        thread_id = self.ident
        if thread_id is not None:
            # 例外を発生させてスレッドを終了させる
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread_id),
                ctypes.py_object(SystemExit),
            )
            if res > 1:
                # 複数のスレッドに例外が送られた場合はリセット
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(thread_id),
                    ctypes.c_long(0),
                )

        self._stop_event.set()

    def stopped(self) -> bool:
        """Check if the thread has received a stop request.

        スレッドが停止要求を受けたかどうかを確認するメソッド.

        Returns:
            bool: True if stop has been requested, False otherwise / 停止要求を受けた場合True、それ以外はFalse
        """
        return self._stop_event.is_set()
