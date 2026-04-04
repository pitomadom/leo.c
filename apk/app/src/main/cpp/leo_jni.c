/*
 * leo_jni.c — JNI bridge for Leo C inference on Android
 *
 * Calls leo.c functions compiled with -DLEO_JNI.
 * Full Gemma-3 inference + Zikharon memory + Dario field.
 *
 * (c) 2026 arianna method
 */

#include <jni.h>
#include <android/log.h>
#include <string.h>
#include <stdlib.h>

#define LOG_TAG "LeoJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

/* Functions from leo.c (compiled with -DLEO_JNI) */
extern int   leo_jni_init(const char *gguf_path, const char *mem_path);
extern char *leo_jni_generate(const char *prompt, int max_tokens);
extern char *leo_jni_dream(void);
extern char *leo_jni_think(void);
extern void  leo_jni_save(void);

JNIEXPORT jboolean JNICALL
Java_com_ariannamethod_leo_LeoEngine_nativeInit(
    JNIEnv *env, jobject thiz, jstring gguf_path, jstring mem_path)
{
    const char *gp = (*env)->GetStringUTFChars(env, gguf_path, NULL);
    const char *mp = (*env)->GetStringUTFChars(env, mem_path, NULL);
    LOGI("Init: gguf=%s mem=%s", gp, mp);

    int ok = leo_jni_init(gp, mp);

    (*env)->ReleaseStringUTFChars(env, gguf_path, gp);
    (*env)->ReleaseStringUTFChars(env, mem_path, mp);

    LOGI("Leo initialized: %s", ok ? "yes" : "no");
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_com_ariannamethod_leo_LeoEngine_nativeGenerate(
    JNIEnv *env, jobject thiz, jstring prompt, jint max_tokens)
{
    const char *p = (*env)->GetStringUTFChars(env, prompt, NULL);
    LOGI("Generate: '%.50s...' max=%d", p, max_tokens);

    char *result = leo_jni_generate(p, (int)max_tokens);

    (*env)->ReleaseStringUTFChars(env, prompt, p);

    jstring jresult = (*env)->NewStringUTF(env, result ? result : "...");
    free(result);
    return jresult;
}

JNIEXPORT jstring JNICALL
Java_com_ariannamethod_leo_LeoEngine_nativeDream(JNIEnv *env, jobject thiz)
{
    char *result = leo_jni_dream();
    jstring jresult = (*env)->NewStringUTF(env, result ? result : "...");
    free(result);
    return jresult;
}

JNIEXPORT jstring JNICALL
Java_com_ariannamethod_leo_LeoEngine_nativeThink(JNIEnv *env, jobject thiz)
{
    char *result = leo_jni_think();
    jstring jresult = (*env)->NewStringUTF(env, result ? result : "...");
    free(result);
    return jresult;
}

JNIEXPORT void JNICALL
Java_com_ariannamethod_leo_LeoEngine_nativeSave(JNIEnv *env, jobject thiz)
{
    LOGI("Saving memory...");
    leo_jni_save();
}

JNIEXPORT jstring JNICALL
Java_com_ariannamethod_leo_LeoEngine_nativeGetStats(JNIEnv *env, jobject thiz)
{
    return (*env)->NewStringUTF(env, "{\"engine\":\"leo.c\",\"arch\":\"gemma3\"}");
}
