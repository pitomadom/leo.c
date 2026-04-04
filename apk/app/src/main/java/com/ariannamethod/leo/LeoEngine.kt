package com.ariannamethod.leo

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Leo Engine — manages GGUF download, native inference via JNI, and Zikharon memory.
 *
 * Lifecycle:
 *   1. First launch: download leo-plain-q8.gguf (278MB) from HuggingFace
 *   2. Load GGUF via JNI (leo.c compiled as .so)
 *   3. Each generate() call: forward pass + Zikharon inject + save memory
 *   4. dream(): Leo thinks without prompt — anchor percolation
 */
class LeoEngine private constructor(private val context: Context) {

    companion object {
        private const val TAG = "LeoEngine"
        private const val GGUF_FILENAME = "leo-1b-plain-q4.gguf"
        private const val GGUF_URL = "https://huggingface.co/ataeff/g/resolve/main/leo-1b-plain-q4.gguf"
        private const val MEM_FILENAME = "leo.mem"

        @Volatile
        private var instance: LeoEngine? = null

        fun getInstance(context: Context): LeoEngine {
            return instance ?: synchronized(this) {
                instance ?: LeoEngine(context.applicationContext).also {
                    instance = it
                    it.init()
                }
            }
        }

        init {
            System.loadLibrary("leo_jni")
        }
    }

    // JNI native methods — bridge to leo.c
    private external fun nativeInit(ggufPath: String, memPath: String): Boolean
    private external fun nativeGenerate(prompt: String, maxTokens: Int): String
    private external fun nativeDream(): String
    private external fun nativeThink(): String
    private external fun nativeSave()
    private external fun nativeGetStats(): String

    private var initialized = false

    private fun init() {
        val ggufFile = File(context.filesDir, GGUF_FILENAME)
        val memFile = File(context.filesDir, MEM_FILENAME)

        if (!ggufFile.exists()) {
            Log.i(TAG, "GGUF not found, will download on first use")
            return
        }

        initialized = nativeInit(ggufFile.absolutePath, memFile.absolutePath)
        Log.i(TAG, "Initialized: $initialized (${ggufFile.length() / 1_000_000}MB)")
    }

    /**
     * Download GGUF weights if not present
     */
    suspend fun ensureWeights(): Boolean = withContext(Dispatchers.IO) {
        val ggufFile = File(context.filesDir, GGUF_FILENAME)
        if (ggufFile.exists() && ggufFile.length() > 100_000_000) {
            return@withContext true
        }

        Log.i(TAG, "Downloading GGUF from $GGUF_URL...")
        try {
            val conn = URL(GGUF_URL).openConnection() as HttpURLConnection
            conn.connectTimeout = 30000
            conn.readTimeout = 60000

            val total = conn.contentLength.toLong()
            val tempFile = File(context.filesDir, "$GGUF_FILENAME.tmp")
            val input = conn.inputStream
            val output = FileOutputStream(tempFile)
            val buffer = ByteArray(65536)
            var downloaded = 0L

            while (true) {
                val n = input.read(buffer)
                if (n < 0) break
                output.write(buffer, 0, n)
                downloaded += n
                if (total > 0 && downloaded % (10 * 1024 * 1024) == 0L) {
                    Log.i(TAG, "Download: ${downloaded / 1_000_000}/${total / 1_000_000} MB")
                }
            }
            output.close()
            input.close()
            conn.disconnect()

            tempFile.renameTo(ggufFile)
            Log.i(TAG, "Download complete: ${ggufFile.length() / 1_000_000} MB")

            // Initialize after download
            val memFile = File(context.filesDir, MEM_FILENAME)
            initialized = nativeInit(ggufFile.absolutePath, memFile.absolutePath)
            return@withContext initialized
        } catch (e: Exception) {
            Log.e(TAG, "Download failed: ${e.message}")
            return@withContext false
        }
    }

    /**
     * Generate response to user prompt
     */
    suspend fun generate(prompt: String): String = withContext(Dispatchers.IO) {
        if (!initialized) {
            if (!ensureWeights()) return@withContext "Leo is downloading weights..."
        }
        try {
            val result = nativeGenerate(prompt, 100)
            nativeSave()
            result
        } catch (e: Exception) {
            Log.e(TAG, "Generate failed: ${e.message}")
            "..."
        }
    }

    /**
     * Dream — Leo thinks without prompt (called every 7 min)
     */
    suspend fun dream(): String = withContext(Dispatchers.IO) {
        if (!initialized) return@withContext "..."
        try {
            nativeDream()
        } catch (e: Exception) {
            "..."
        }
    }

    /**
     * Think — Leo generates from internal state (widget refresh)
     */
    suspend fun think(): String = withContext(Dispatchers.IO) {
        if (!initialized) {
            if (!ensureWeights()) return@withContext "Leo awakens..."
        }
        try {
            nativeThink()
        } catch (e: Exception) {
            "..."
        }
    }
}
