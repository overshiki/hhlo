/* cbits/pjrt_shim.c
 * Minimal C shim around the PJRT C API.
 * We include the full upstream pjrt_c_api.h and expose
 * a small set of wrapper functions with simpler signatures.
 */

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include "pjrt_c_api.h"
#include "pjrt_shim.h"

// ---------------------------------------------------------------------------
// Plugin loading
// ---------------------------------------------------------------------------

PJRT_Error* hhlo_pjrt_load_plugin(const char* path, PJRT_Api** out_api) {
    void* handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        *out_api = NULL;
        return NULL;
    }

    PJRT_Api* (*get_api)(void) = (PJRT_Api* (*)(void)) dlsym(handle, "GetPjrtApi");
    if (!get_api) {
        dlclose(handle);
        *out_api = NULL;
        return NULL;
    }

    *out_api = get_api();
    return NULL;
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

PJRT_Error* hhlo_pjrt_create_client(PJRT_Api* api, PJRT_Client** out_client) {
    PJRT_Client_Create_Args args = {0};
    args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    args.client = NULL;

    PJRT_Error* err = api->PJRT_Client_Create(&args);
    if (err == NULL) {
        *out_client = args.client;
    }
    return err;
}

PJRT_Error* hhlo_pjrt_client_destroy(PJRT_Api* api, PJRT_Client* client) {
    PJRT_Client_Destroy_Args args = {0};
    args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    args.client = client;
    return api->PJRT_Client_Destroy(&args);
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

// Encode a uint64 as a protobuf varint. Returns number of bytes written.
static size_t encode_varint(uint64_t value, char* out) {
    size_t i = 0;
    while (value >= 0x80) {
        out[i++] = (char)((value & 0x7f) | 0x80);
        value >>= 7;
    }
    out[i++] = (char)value;
    return i;
}

// Build a minimal CompileOptionsProto with configurable num_replicas.
// The proto structure is:
//   message CompileOptions {
//     ExecutableBuildOptions executable_build_options = 3;
//   }
//   message ExecutableBuildOptions {
//     int32 num_replicas   = 4;
//     int32 num_partitions = 5;
//   }
static size_t build_compile_options_proto(int num_replicas, char* out, size_t out_size) {
    // We need: 0x1a [len] [0x20 replicas_varint] [0x28 0x01]
    // Max varint length for int32 is 5 bytes, but for small values (<=127) it's 1.
    char replicas_varint[5];
    size_t replicas_len = encode_varint((uint64_t)num_replicas, replicas_varint);

    size_t submsg_len = 1 + replicas_len + 2;  // tag4 + varint + tag5 + 0x01
    size_t total_len = 1 + 1 + submsg_len;     // 0x1a + len_byte + submsg

    if (total_len > out_size) return 0;

    size_t i = 0;
    out[i++] = 0x1a;                    // field 3, wire type 2 (length-delimited)
    out[i++] = (char)submsg_len;        // submessage length (fits in 1 byte for small values)
    out[i++] = 0x20;                    // field 4, wire type 0 (varint)
    for (size_t j = 0; j < replicas_len; ++j) {
        out[i++] = replicas_varint[j];
    }
    out[i++] = 0x28;                    // field 5, wire type 0 (varint)
    out[i++] = 0x01;                    // num_partitions = 1

    return i;
}

PJRT_Error* hhlo_pjrt_compile(PJRT_Api* api, PJRT_Client* client,
                               const char* code, size_t code_size,
                               PJRT_LoadedExecutable** out_exec) {
    return hhlo_pjrt_compile_with_options(api, client, code, code_size, 1, out_exec);
}

PJRT_Error* hhlo_pjrt_compile_with_options(PJRT_Api* api, PJRT_Client* client,
                                            const char* code, size_t code_size,
                                            int num_replicas,
                                            PJRT_LoadedExecutable** out_exec) {
    PJRT_Program program = {0};
    program.struct_size = PJRT_Program_STRUCT_SIZE;
    program.code = (char*) code;
    program.code_size = code_size;
    program.format = "mlir";
    program.format_size = 4;

    char compile_options_proto[16];
    size_t proto_size = build_compile_options_proto(num_replicas, compile_options_proto, sizeof(compile_options_proto));

    PJRT_Client_Compile_Args args = {0};
    args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    args.client = client;
    args.program = &program;
    args.compile_options = compile_options_proto;
    args.compile_options_size = proto_size;
    args.executable = NULL;

    PJRT_Error* err = api->PJRT_Client_Compile(&args);
    if (err == NULL) {
        *out_exec = args.executable;
    }
    return err;
}

PJRT_Error* hhlo_pjrt_loaded_executable_destroy(PJRT_Api* api,
                                                  PJRT_LoadedExecutable* exec) {
    PJRT_LoadedExecutable_Destroy_Args args = {0};
    args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    args.executable = exec;
    return api->PJRT_LoadedExecutable_Destroy(&args);
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

PJRT_Error* hhlo_pjrt_execute(PJRT_Api* api, PJRT_LoadedExecutable* exec,
                               size_t num_args, PJRT_Buffer** args_in,
                               size_t max_outputs,
                               PJRT_Buffer** out_outputs,
                               size_t* out_num_outputs) {
    // Single-device execution for simplicity
    PJRT_ExecuteOptions options = {0};
    options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

    PJRT_Buffer* const* arg_list = (PJRT_Buffer* const*) args_in;

    // Output pre-allocation: caller provides array of PJRT_Buffer* of size max_outputs
    PJRT_Buffer** output_list = out_outputs;

    PJRT_LoadedExecutable_Execute_Args exec_args = {0};
    exec_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    exec_args.executable = exec;
    exec_args.options = &options;
    exec_args.argument_lists = &arg_list;
    exec_args.num_devices = 1;
    exec_args.num_args = num_args;
    exec_args.output_lists = &output_list;
    exec_args.device_complete_events = NULL;
    exec_args.execute_device = NULL;

    PJRT_Error* err = api->PJRT_LoadedExecutable_Execute(&exec_args);
    if (err == NULL) {
        // Count outputs by finding how many non-NULL entries were written
        size_t n = 0;
        for (size_t i = 0; i < max_outputs; ++i) {
            if (out_outputs[i] != NULL) n++;
            else break;
        }
        *out_num_outputs = n;
    }
    return err;
}

// ---------------------------------------------------------------------------
// Buffer type constants (exposed to Haskell FFI)
// ---------------------------------------------------------------------------

int hhlo_buffer_type_invalid(void)   { return PJRT_Buffer_Type_INVALID; }
int hhlo_buffer_type_pred(void)      { return PJRT_Buffer_Type_PRED; }
int hhlo_buffer_type_s8(void)        { return PJRT_Buffer_Type_S8; }
int hhlo_buffer_type_s16(void)       { return PJRT_Buffer_Type_S16; }
int hhlo_buffer_type_s32(void)       { return PJRT_Buffer_Type_S32; }
int hhlo_buffer_type_s64(void)       { return PJRT_Buffer_Type_S64; }
int hhlo_buffer_type_u8(void)        { return PJRT_Buffer_Type_U8; }
int hhlo_buffer_type_u16(void)       { return PJRT_Buffer_Type_U16; }
int hhlo_buffer_type_u32(void)       { return PJRT_Buffer_Type_U32; }
int hhlo_buffer_type_u64(void)       { return PJRT_Buffer_Type_U64; }
int hhlo_buffer_type_f16(void)       { return PJRT_Buffer_Type_F16; }
int hhlo_buffer_type_f32(void)       { return PJRT_Buffer_Type_F32; }
int hhlo_buffer_type_f64(void)       { return PJRT_Buffer_Type_F64; }
int hhlo_buffer_type_bf16(void)      { return PJRT_Buffer_Type_BF16; }
int hhlo_buffer_type_c64(void)       { return PJRT_Buffer_Type_C64; }
int hhlo_buffer_type_c128(void)      { return PJRT_Buffer_Type_C128; }

// ---------------------------------------------------------------------------
// Executable metadata
// ---------------------------------------------------------------------------

PJRT_Error* hhlo_pjrt_executable_num_outputs(PJRT_Api* api,
                                              PJRT_LoadedExecutable* loaded_exec,
                                              size_t* out_num_outputs) {
    // Get the underlying PJRT_Executable
    PJRT_LoadedExecutable_GetExecutable_Args get_args = {0};
    get_args.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
    get_args.loaded_executable = loaded_exec;
    get_args.executable = NULL;

    PJRT_Error* err = api->PJRT_LoadedExecutable_GetExecutable(&get_args);
    if (err != NULL) {
        return err;
    }

    // Query number of outputs
    PJRT_Executable_NumOutputs_Args num_args = {0};
    num_args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
    num_args.executable = get_args.executable;
    err = api->PJRT_Executable_NumOutputs(&num_args);
    if (err != NULL) {
        api->PJRT_Executable_Destroy(&(PJRT_Executable_Destroy_Args){
            .struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE,
            .executable = get_args.executable
        });
        return err;
    }

    *out_num_outputs = num_args.num_outputs;

    // Clean up the temporary PJRT_Executable
    api->PJRT_Executable_Destroy(&(PJRT_Executable_Destroy_Args){
        .struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE,
        .executable = get_args.executable
    });

    return NULL;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static PJRT_Device* get_first_addressable_device(PJRT_Api* api, PJRT_Client* client) {
    PJRT_Client_AddressableDevices_Args args = {0};
    args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    args.client = client;
    PJRT_Error* err = api->PJRT_Client_AddressableDevices(&args);
    if (err != NULL || args.num_addressable_devices == 0) {
        if (err) api->PJRT_Error_Destroy(&(PJRT_Error_Destroy_Args){.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE, .error = err});
        return NULL;
    }
    return args.addressable_devices[0];
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

PJRT_Error* hhlo_pjrt_buffer_from_host(PJRT_Api* api, PJRT_Client* client,
                                        const void* data,
                                        PJRT_Buffer_Type type,
                                        const int64_t* dims, size_t num_dims,
                                        PJRT_Buffer** out_buffer) {
    PJRT_Device* device = get_first_addressable_device(api, client);

    PJRT_Client_BufferFromHostBuffer_Args args = {0};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = client;
    args.data = data;
    args.type = type;
    args.dims = dims;
    args.num_dims = num_dims;
    args.byte_strides = NULL;
    args.num_byte_strides = 0;
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    args.device = device;
    args.memory = NULL;
    args.device_layout = NULL;
    args.done_with_host_buffer = NULL;
    args.buffer = NULL;

    PJRT_Error* err = api->PJRT_Client_BufferFromHostBuffer(&args);
    if (err == NULL) {
        *out_buffer = args.buffer;
    }
    return err;
}

PJRT_Error* hhlo_pjrt_buffer_to_host(PJRT_Api* api, PJRT_Buffer* buffer,
                                      void* dst, size_t dst_size,
                                      PJRT_Event** out_event) {
    PJRT_Buffer_ToHostBuffer_Args args = {0};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = buffer;
    args.host_layout = NULL;
    args.dst = dst;
    args.dst_size = dst_size;
    args.event = NULL;

    PJRT_Error* err = api->PJRT_Buffer_ToHostBuffer(&args);
    if (err == NULL && args.event != NULL) {
        PJRT_Event_Await_Args await_args = {0};
        await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        await_args.event = args.event;
        api->PJRT_Event_Await(&await_args);
        PJRT_Event_Destroy_Args destroy_args = {0};
        destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        destroy_args.event = args.event;
        api->PJRT_Event_Destroy(&destroy_args);
    }
    if (err == NULL && out_event != NULL) {
        *out_event = args.event;
    }
    return err;
}

PJRT_Error* hhlo_pjrt_buffer_destroy(PJRT_Api* api, PJRT_Buffer* buffer) {
    PJRT_Buffer_Destroy_Args args = {0};
    args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    args.buffer = buffer;
    return api->PJRT_Buffer_Destroy(&args);
}

PJRT_Error* hhlo_pjrt_buffer_dimensions(PJRT_Api* api, PJRT_Buffer* buffer,
                                         const int64_t** out_dims, size_t* out_num_dims) {
    PJRT_Buffer_Dimensions_Args args = {0};
    args.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
    args.buffer = buffer;
    PJRT_Error* err = api->PJRT_Buffer_Dimensions(&args);
    if (err == NULL) {
        *out_dims = args.dims;
        *out_num_dims = args.num_dims;
    }
    return err;
}

PJRT_Error* hhlo_pjrt_buffer_element_type(PJRT_Api* api, PJRT_Buffer* buffer, int* out_type) {
    PJRT_Buffer_ElementType_Args args = {0};
    args.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
    args.buffer = buffer;
    PJRT_Error* err = api->PJRT_Buffer_ElementType(&args);
    if (err == NULL) {
        *out_type = (int) args.type;
    }
    return err;
}

PJRT_Error* hhlo_pjrt_buffer_on_device_size(PJRT_Api* api, PJRT_Buffer* buffer, size_t* out_size) {
    PJRT_Buffer_OnDeviceSizeInBytes_Args args = {0};
    args.struct_size = PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE;
    args.buffer = buffer;
    PJRT_Error* err = api->PJRT_Buffer_OnDeviceSizeInBytes(&args);
    if (err == NULL) {
        *out_size = args.on_device_size_in_bytes;
    }
    return err;
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

PJRT_Error* hhlo_pjrt_buffer_ready_event(PJRT_Api* api, PJRT_Buffer* buffer,
                                                PJRT_Event** out_event) {
    PJRT_Buffer_ReadyEvent_Args args = {0};
    args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    args.buffer = buffer;
    args.event = NULL;

    PJRT_Error* err = api->PJRT_Buffer_ReadyEvent(&args);
    if (err == NULL) {
        *out_event = args.event;
    }
    return err;
}

PJRT_Error* hhlo_pjrt_event_is_ready(PJRT_Api* api, PJRT_Event* event, int* out_ready) {
    PJRT_Event_IsReady_Args args = {0};
    args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
    args.event = event;
    PJRT_Error* err = api->PJRT_Event_IsReady(&args);
    if (err == NULL) {
        *out_ready = args.is_ready ? 1 : 0;
    }
    return err;
}

PJRT_Error* hhlo_pjrt_event_await(PJRT_Api* api, PJRT_Event* event) {
    PJRT_Event_Await_Args args = {0};
    args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    args.event = event;
    return api->PJRT_Event_Await(&args);
}

PJRT_Error* hhlo_pjrt_event_destroy(PJRT_Api* api, PJRT_Event* event) {
    PJRT_Event_Destroy_Args args = {0};
    args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    args.event = event;
    return api->PJRT_Event_Destroy(&args);
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

PJRT_Error* hhlo_pjrt_error_message(PJRT_Api* api, PJRT_Error* error,
                                     const char** out_msg, size_t* out_size) {
    PJRT_Error_Message_Args args = {0};
    args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    args.error = error;
    args.message = NULL;
    args.message_size = 0;

    api->PJRT_Error_Message(&args);
    *out_msg = args.message;
    *out_size = args.message_size;
    return NULL;
}

PJRT_Error* hhlo_pjrt_error_destroy(PJRT_Api* api, PJRT_Error* error) {
    PJRT_Error_Destroy_Args args = {0};
    args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    args.error = error;
    api->PJRT_Error_Destroy(&args);
    return NULL;
}

// ---------------------------------------------------------------------------
// Device enumeration
// ---------------------------------------------------------------------------

PJRT_Error* hhlo_pjrt_client_addressable_device_count(PJRT_Api* api,
                                                       PJRT_Client* client,
                                                       size_t* out_count) {
    PJRT_Client_AddressableDevices_Args args = {0};
    args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    args.client = client;
    PJRT_Error* err = api->PJRT_Client_AddressableDevices(&args);
    if (err == NULL) {
        *out_count = args.num_addressable_devices;
    }
    return err;
}

PJRT_Error* hhlo_pjrt_client_addressable_device(PJRT_Api* api,
                                                 PJRT_Client* client,
                                                 size_t index,
                                                 PJRT_Device** out_device) {
    PJRT_Client_AddressableDevices_Args args = {0};
    args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    args.client = client;
    PJRT_Error* err = api->PJRT_Client_AddressableDevices(&args);
    if (err != NULL) {
        return err;
    }
    if (index >= args.num_addressable_devices) {
        *out_device = NULL;
        return NULL;
    }
    *out_device = args.addressable_devices[index];
    return NULL;
}

PJRT_Error* hhlo_pjrt_device_id(PJRT_Api* api, PJRT_Device* device,
                                 int* out_id) {
    PJRT_Device_GetDescription_Args desc_args = {0};
    desc_args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
    desc_args.device = device;
    PJRT_Error* err = api->PJRT_Device_GetDescription(&desc_args);
    if (err != NULL) {
        return err;
    }

    PJRT_DeviceDescription_Id_Args id_args = {0};
    id_args.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
    id_args.device_description = desc_args.device_description;
    err = api->PJRT_DeviceDescription_Id(&id_args);
    if (err == NULL) {
        *out_id = (int) id_args.id;
    }
    return err;
}

PJRT_Error* hhlo_pjrt_device_kind(PJRT_Api* api, PJRT_Device* device,
                                   const char** out_kind,
                                   size_t* out_kind_len) {
    PJRT_Device_GetDescription_Args desc_args = {0};
    desc_args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
    desc_args.device = device;
    PJRT_Error* err = api->PJRT_Device_GetDescription(&desc_args);
    if (err != NULL) {
        return err;
    }

    PJRT_DeviceDescription_Kind_Args kind_args = {0};
    kind_args.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
    kind_args.device_description = desc_args.device_description;
    err = api->PJRT_DeviceDescription_Kind(&kind_args);
    if (err == NULL) {
        *out_kind = kind_args.device_kind;
        *out_kind_len = kind_args.device_kind_size;
    }
    return err;
}

// ---------------------------------------------------------------------------
// Device-aware buffer creation
// ---------------------------------------------------------------------------

PJRT_Error* hhlo_pjrt_buffer_from_host_on_device(PJRT_Api* api,
                                                  PJRT_Client* client,
                                                  PJRT_Device* device,
                                                  const void* data,
                                                  PJRT_Buffer_Type type,
                                                  const int64_t* dims,
                                                  size_t num_dims,
                                                  PJRT_Buffer** out_buffer) {
    PJRT_Client_BufferFromHostBuffer_Args args = {0};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = client;
    args.data = data;
    args.type = type;
    args.dims = dims;
    args.num_dims = num_dims;
    args.byte_strides = NULL;
    args.num_byte_strides = 0;
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    args.device = device;
    args.memory = NULL;
    args.device_layout = NULL;
    args.done_with_host_buffer = NULL;
    args.buffer = NULL;

    PJRT_Error* err = api->PJRT_Client_BufferFromHostBuffer(&args);
    if (err == NULL) {
        *out_buffer = args.buffer;
    }
    return err;
}

// ---------------------------------------------------------------------------
// Async D2H
// ---------------------------------------------------------------------------

PJRT_Error* hhlo_pjrt_buffer_to_host_async(PJRT_Api* api, PJRT_Buffer* buffer,
                                            void* dst, size_t dst_size,
                                            PJRT_Event** out_event) {
    PJRT_Buffer_ToHostBuffer_Args args = {0};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = buffer;
    args.host_layout = NULL;
    args.dst = dst;
    args.dst_size = dst_size;
    args.event = NULL;

    PJRT_Error* err = api->PJRT_Buffer_ToHostBuffer(&args);
    if (err == NULL && out_event != NULL) {
        *out_event = args.event;
    }
    return err;
}

// ---------------------------------------------------------------------------
// Device-aware execution
// ---------------------------------------------------------------------------

PJRT_Error* hhlo_pjrt_execute_on_device(PJRT_Api* api,
                                         PJRT_LoadedExecutable* exec,
                                         size_t num_args, PJRT_Buffer** args_in,
                                         PJRT_Device* execute_device,
                                         size_t max_outputs,
                                         PJRT_Buffer** out_outputs,
                                         size_t* out_num_outputs) {
    PJRT_ExecuteOptions options = {0};
    options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

    PJRT_Buffer* const* arg_list = (PJRT_Buffer* const*) args_in;
    PJRT_Buffer** output_list = out_outputs;

    PJRT_LoadedExecutable_Execute_Args exec_args = {0};
    exec_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    exec_args.executable = exec;
    exec_args.options = &options;
    exec_args.argument_lists = &arg_list;
    exec_args.num_devices = 1;
    exec_args.num_args = num_args;
    exec_args.output_lists = &output_list;
    exec_args.device_complete_events = NULL;
    exec_args.execute_device = execute_device;

    PJRT_Error* err = api->PJRT_LoadedExecutable_Execute(&exec_args);
    if (err == NULL) {
        size_t n = 0;
        for (size_t i = 0; i < max_outputs; ++i) {
            if (out_outputs[i] != NULL) n++;
            else break;
        }
        *out_num_outputs = n;
    }
    return err;
}

PJRT_Error* hhlo_pjrt_execute_multi(PJRT_Api* api,
                                     PJRT_LoadedExecutable* exec,
                                     size_t num_devices,
                                     size_t num_args,
                                     PJRT_Buffer*** args_in,
                                     size_t max_outputs,
                                     PJRT_Buffer*** out_outputs,
                                     size_t* out_num_outputs_per_device) {
    PJRT_ExecuteOptions options = {0};
    options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

    PJRT_LoadedExecutable_Execute_Args exec_args = {0};
    exec_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    exec_args.executable = exec;
    exec_args.options = &options;
    exec_args.argument_lists = (PJRT_Buffer* const* const*) args_in;
    exec_args.num_devices = num_devices;
    exec_args.num_args = num_args;
    exec_args.output_lists = (PJRT_Buffer** const*) out_outputs;
    exec_args.device_complete_events = NULL;
    exec_args.execute_device = NULL;

    PJRT_Error* err = api->PJRT_LoadedExecutable_Execute(&exec_args);
    if (err == NULL) {
        for (size_t d = 0; d < num_devices; ++d) {
            size_t n = 0;
            for (size_t i = 0; i < max_outputs; ++i) {
                if (out_outputs[d][i] != NULL) n++;
                else break;
            }
            out_num_outputs_per_device[d] = n;
        }
    }
    return err;
}

// ---------------------------------------------------------------------------
// Custom calls (GPU)
// ---------------------------------------------------------------------------

// Local definitions for PJRT GPU Custom Call extension
// (matches xla/pjrt/c/pjrt_c_api_gpu_extension.h)

typedef struct {
    size_t struct_size;
    const char* function_name;
    size_t function_name_size;
    int api_version;
    void* handler_instantiate;
    void* handler_prepare;
    void* handler_initialize;
    void* handler_execute;
} PJRT_Gpu_Register_Custom_Call_Args;

typedef PJRT_Error* PJRT_Gpu_Register_Custom_Call(
    PJRT_Gpu_Register_Custom_Call_Args* args);

typedef struct {
    PJRT_Extension_Base base;
    PJRT_Gpu_Register_Custom_Call* custom_call;
} PJRT_Gpu_Custom_Call;

// Load a shared library, look up 'function_name', and register it with the
// PJRT GPU plugin via the PJRT_Gpu_Custom_Call extension.
//
// Returns 0 on success, or a negative code on failure.
// On failure, *out_error_msg is set to a static/dynamic error string.
int hhlo_pjrt_register_gpu_custom_call(PJRT_Api* api, const char* lib_path,
                                        const char* function_name,
                                        const char** out_error_msg) {
    *out_error_msg = NULL;

    // 1. Load the library (keep it open for the process lifetime)
    void* handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        *out_error_msg = dlerror();
        return -1;
    }

    // 2. Look up the target symbol
    void* symbol = dlsym(handle, function_name);
    if (!symbol) {
        *out_error_msg = dlerror();
        return -2;
    }

    // 3. Walk the PJRT extension chain to find the GPU custom call extension
    PJRT_Extension_Base* ext = api->extension_start;
    while (ext != NULL) {
        if (ext->type == PJRT_Extension_Type_Gpu_Custom_Call) {
            PJRT_Gpu_Custom_Call* gpu_ext = (PJRT_Gpu_Custom_Call*)ext;

            PJRT_Gpu_Register_Custom_Call_Args args = {0};
            args.struct_size = sizeof(PJRT_Gpu_Register_Custom_Call_Args);
            args.function_name = function_name;
            args.function_name_size = strlen(function_name);
            args.api_version = 0;        // 0 = untyped / original ABI
            args.handler_execute = symbol;

            PJRT_Error* err = gpu_ext->custom_call(&args);
            if (err != NULL) {
                *out_error_msg = "PJRT_Gpu_Register_Custom_Call returned an error";
                return -3;
            }
            return 0;
        }
        ext = ext->next;
    }

    *out_error_msg = "PJRT GPU custom call extension not found";
    return -4;
}
