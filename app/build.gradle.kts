import java.util.Properties
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.kotlin.plugin.serialization")
}

android {
    namespace = "io.github.derweh.bayesianbahn"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.github.derweh.bayesianbahn"
        minSdk = 26
        targetSdk = 35
        versionCode = 7
        versionName = "0.3.0"
    }

    // Release signing from an untracked keystore.properties (or CI secrets);
    // absent -> unsigned build (F-Droid signs with its own key).
    val keystoreProperties = rootProject.file("keystore.properties")
    if (keystoreProperties.exists()) {
        val props = Properties()
        keystoreProperties.inputStream().use { props.load(it) }
        signingConfigs {
            create("release") {
                storeFile = rootProject.file(props.getProperty("storeFile"))
                storePassword = props.getProperty("storePassword")
                keyAlias = props.getProperty("keyAlias")
                keyPassword = props.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
            signingConfig = signingConfigs.findByName("release")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
    }
    // Reproducible builds for F-Droid: strip baseline profiles and versioned dex metadata
    dependenciesInfo {
        includeInApk = false
        includeInBundle = false
    }
    packaging {
        resources.excludes += setOf("META-INF/AL2.0", "META-INF/LGPL2.1")
    }
}

// TranslationCompletenessTest reads res/ as files rather than through the
// generated R class, so Gradle does not know a changed string can change the
// result: without this the task reports UP-TO-DATE and a stale pass.
tasks.withType<Test>().configureEach {
    inputs.dir(layout.projectDirectory.dir("src/main/res"))
        .withPathSensitivity(PathSensitivity.RELATIVE)
        .withPropertyName("resources")

    // The evaluation harnesses are driven entirely by HARNESS_* environment
    // variables — which day, which events, where the answers go — and Gradle
    // cannot see those, so a second run with a different day reported
    // UP-TO-DATE and left the first day's answers in place. The driver worked
    // around it with --rerun-tasks, which reruns *every* task: the app was
    // recompiled from scratch before each of the six harness passes per day,
    // turning a one-second test into a thirty-second one. Declaring the
    // variables as inputs invalidates the test alone and leaves the build
    // cached.
    inputs.properties(providers.environmentVariablesPrefixedBy("HARNESS_").get())
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")

    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    testImplementation("junit:junit:4.13.2")
    // Same artifact group as the OkHttp already used, so no new upstream: it
    // is the only way to test the HTTP cache against real responses, headers
    // and 304s rather than against a mock of our own beliefs about them.
    testImplementation("com.squareup.okhttp3:mockwebserver:4.12.0")
    testImplementation("org.mockito:mockito-core:5.14.2")
    testImplementation("net.sf.kxml:kxml2:2.3.0")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
}
