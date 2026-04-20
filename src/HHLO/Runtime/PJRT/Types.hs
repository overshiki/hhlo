module HHLO.Runtime.PJRT.Types
    ( PJRTApi(..)
    , PJRTClient(..)
    , PJRTBuffer(..)
    , PJRTExecutable(..)
    , PJRTError(..)
    , PJRTEvent(..)
    ) where

import Foreign.Ptr

newtype PJRTApi        = PJRTApi        (Ptr PJRTApi)
newtype PJRTClient     = PJRTClient     (Ptr PJRTClient)
newtype PJRTBuffer     = PJRTBuffer     (Ptr PJRTBuffer)
newtype PJRTExecutable = PJRTExecutable (Ptr PJRTExecutable)
newtype PJRTError      = PJRTError      (Ptr PJRTError)
newtype PJRTEvent      = PJRTEvent      (Ptr PJRTEvent)
